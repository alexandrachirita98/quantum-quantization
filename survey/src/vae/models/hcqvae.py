import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def amplitude_embed(x):
    """L2-normalize a real (B, 2**n) vector and read it as a batch of pure-state amplitudes."""
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x.to(torch.complex64)

def phase(a):
    return torch.complex(torch.cos(a), torch.sin(a))  # e^{i a}, differentiable in a

def rot_matrix(phi, theta, omega):
    """qml.Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi), a (2, 2) unitary."""
    c = torch.cos(theta / 2).to(torch.complex64)
    s = torch.sin(theta / 2).to(torch.complex64)
    m00 = phase(-(phi + omega) / 2) * c
    m01 = -phase((phi - omega) / 2) * s
    m10 = phase(-(phi - omega) / 2) * s
    m11 = phase((phi + omega) / 2) * c
    return torch.stack([torch.stack([m00, m01]), torch.stack([m10, m11])])

def apply_1q(psi, G, q, n):
    """Apply a single-qubit gate G to wire q of a batch of statevectors psi (B, 2**n)."""
    B = psi.shape[0]
    psi = psi.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
    psi = torch.einsum('xy,bpyz->bpxz', G, psi)  # contract the target qubit's axis
    return psi.reshape(B, 2 ** n)

def cnot_perm(control, target, n, device):
    """Amplitude permutation implementing CNOT(control, target); wire 0 is the most significant bit."""
    idx = torch.arange(2 ** n, device=device)
    control_bit = (idx >> (n - 1 - control)) & 1
    target_mask = 1 << (n - 1 - target)
    return torch.where(control_bit.bool(), idx ^ target_mask, idx)

def strongly_entangling(psi, weights, n):
    """PennyLane StronglyEntanglingLayers: per-qubit Rot then a CNOT ring, per layer."""
    n_layers = weights.shape[0]
    ranges = [(l % (n - 1)) + 1 for l in range(n_layers)]  # PennyLane default ranges
    for l in range(n_layers):
        for i in range(n):
            G = rot_matrix(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2])
            psi = apply_1q(psi, G, i, n)
        r = ranges[l]
        for i in range(n):
            psi = psi[:, cnot_perm(i, (i + r) % n, n, psi.device)]
    return psi


WIRES = 10  # 10-qubit circuit, matching the repo

def measure_z(psi, n):
    """Pauli-Z expectation on each wire: (B, 2**n) statevectors -> (B, n) values in [-1, 1]."""
    probs = (psi.conj() * psi).real
    idx = torch.arange(2 ** n, device=psi.device)
    signs = torch.stack([1 - 2 * ((idx >> (n - 1 - j)) & 1) for j in range(n)]).float()
    return probs @ signs.t()

class QuantumLayer(nn.Module):
    """AmplitudeEmbedding -> StronglyEntanglingLayers -> <Z> per wire; the torch analogue of
    qml.qnn.TorchLayer(circuit, {"weights": StronglyEntanglingLayers.shape(3, WIRES)})."""
    def __init__(self, n_wires=WIRES, n_layers=3):
        super().__init__()
        self.n = n_wires
        # PennyLane TorchLayer default init: uniform on [0, 2*pi)
        self.weights = nn.Parameter(torch.empty(n_layers, n_wires, 3).uniform_(0, 2 * math.pi))

    def forward(self, x):
        psi = amplitude_embed(x)
        psi = strongly_entangling(psi, self.weights, self.n)
        return measure_z(psi, self.n)


TOPIC_COUNT = 20  # number of latent "topics", matching the repo

class HCQVAE(nn.Module):
    def __init__(self, vocab_size, n_wires=WIRES, n_topics=TOPIC_COUNT):
        super().__init__()
        self.dropout = nn.Dropout(p=.25)
        self.encoder_fc1024 = nn.Linear(vocab_size, 2 ** n_wires)
        self.encoder_fc10_mu = QuantumLayer(n_wires, 3)
        self.encoder_fc10_logvar = QuantumLayer(n_wires, 3)
        self.gsm_fc = nn.Linear(n_wires, n_topics)

        self.mu_alpha = nn.Parameter(torch.randn(n_wires))
        self.log_var_alpha = nn.Parameter(torch.randn(n_wires))
        self.bn_mu = nn.BatchNorm1d(n_wires, affine=False)
        self.bn_log_var = nn.BatchNorm1d(n_wires, affine=False)
        self.temperature = nn.Parameter(torch.randn(1).squeeze())

        # classical embedding-matrix decoder (randomly initialized; the repo seeds word_embeddings
        # from GloVe-300, which has no analogue for pixels)
        self.word_embeddings = nn.Linear(300, vocab_size, bias=False)
        self.topic_embeddings = nn.Linear(n_topics, 300, bias=False)

    def encode(self, batch):
        batch = self.dropout(batch)
        batch = F.tanh(self.encoder_fc1024(batch))
        return (self.bn_mu(self.mu_alpha * self.encoder_fc10_mu(batch)),
                self.bn_log_var(self.log_var_alpha * self.encoder_fc10_logvar(batch)))

    def decode(self, batch):
        return F.softmax(self.word_embeddings(self.topic_embeddings(batch)), dim=-1)

    def get_topics(self):
        return F.softmax(self.word_embeddings(self.topic_embeddings.weight.t()), dim=-1)

    def get_topic_embeddings(self):
        return self.topic_embeddings.weight.t()

    def gsm(self, z):
        """Gaussian-softmax: map the Gaussian latent to a distribution over topics."""
        return F.softmax(self.temperature * self.gsm_fc(z), dim=-1)

    def forward(self, batch):
        mu, log_var = self.encode(batch)
        sigma = log_var.exp().sqrt()
        eta = torch.randn_like(mu) * sigma + mu  # reparameterization
        z = self.gsm(eta)
        return self.decode(z), mu, sigma.log()
