import math

import torch
import torch.nn as nn


def apply_1q(psi, gate, q, n):
    """Apply a single-qubit gate to qubit q of a batched statevector.

    psi: (B, 2^n) complex. gate: (2, 2) shared across the batch, or (B, 2, 2) per-sample
    (used for the data-dependent feature-map rotations). Contracts the size-2 axis for qubit q.
    """
    b = psi.size(0)
    left, right = 2 ** q, 2 ** (n - q - 1)
    psi = psi.reshape(b, left, 2, right)
    if gate.dim() == 2:
        out = torch.einsum('ij,bljr->blir', gate, psi)
    else:
        out = torch.einsum('bij,bljr->blir', gate, psi)
    return out.reshape(b, -1)

# CX on the 4-dim (control, target) block, control = high bit: |10>->|11>, |11>->|10>
CX = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex128)

def apply_cx(psi, ctrl, tgt, n):
    """CX on adjacent qubits with ctrl < tgt (linear entanglement)."""
    b = psi.size(0)
    left, right = 2 ** ctrl, 2 ** (n - tgt - 1)
    psi = psi.reshape(b, left, 4, right)
    out = torch.einsum('ij,bljr->blir', CX.to(psi.device), psi)
    return out.reshape(b, -1)

H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)

def phase_gate(theta):
    """diag(1, e^{i*theta}), batched over theta: (B,) -> (B, 2, 2)."""
    b = theta.size(0)
    g = torch.zeros(b, 2, 2, dtype=torch.complex128, device=theta.device)
    g[:, 0, 0] = 1.0
    g[:, 1, 1] = torch.exp(1j * theta)
    return g

def ry_gate(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)

def rx_gate(theta):
    c = torch.cos(theta / 2).to(torch.complex128)
    s = torch.sin(theta / 2).to(torch.complex128)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])

def z_feature_map(psi, x, n, reps):
    """ZFeatureMap: `reps` blocks of [H on every qubit, P(2*x_i) on qubit i]. x: (B, n) angles."""
    for _ in range(reps):
        for q in range(n):
            psi = apply_1q(psi, H.to(psi.device), q, n)
        for q in range(n):
            psi = apply_1q(psi, phase_gate(2.0 * x[:, q]), q, n)
    return psi

def two_local(psi, theta, n, reps):
    """TwoLocal ry/rx + linear cx. `reps` entangling layers, `reps + 1` rotation layers
    (2*n params each: ry on every qubit then rx on every qubit)."""
    idx = 0
    for r in range(reps + 1):
        for q in range(n):
            psi = apply_1q(psi, ry_gate(theta[idx]), q, n); idx += 1
        for q in range(n):
            psi = apply_1q(psi, rx_gate(theta[idx]), q, n); idx += 1
        if r < reps:
            for q in range(n - 1):
                psi = apply_cx(psi, q, q + 1, n)
    return psi

def num_ansatz_params(n, reps):
    return 2 * n * (reps + 1)

def decoder_distribution(angles, theta, n, fm_reps, ansatz_reps):
    """angles: (B, n) preprocessed input -> (B, 2^n) probability distribution over bitstrings."""
    b = angles.size(0)
    psi = torch.zeros(b, 2 ** n, dtype=torch.complex128, device=angles.device)
    psi[:, 0] = 1.0  # |0...0>
    psi = z_feature_map(psi, angles, n, fm_reps)
    psi = two_local(psi, theta, n, ansatz_reps)
    return psi.abs() ** 2


class QeVAE(nn.Module):
    def __init__(self, n_qubit=6, latent_dim=4, fm_reps=2, ansatz_reps=2):
        super().__init__()
        self.n_qubit = n_qubit
        self.latent_dim = latent_dim
        self.fm_reps = fm_reps
        self.ansatz_reps = ansatz_reps

        self.encoder = nn.Sequential(
            nn.Linear(n_qubit, 8), nn.LeakyReLU(0.01),
            nn.Linear(8, 7), nn.LeakyReLU(0.01))
        self.z_mean = nn.Linear(7, latent_dim)
        self.z_log_var = nn.Linear(7, latent_dim)

        self.preprocessor = nn.Linear(latent_dim, n_qubit)
        nn.init.normal_(self.preprocessor.weight, mean=0, std=0.01)
        nn.init.constant_(self.preprocessor.bias, val=0)

        self.decoder_theta = nn.Parameter(0.1 * torch.randn(num_ansatz_params(n_qubit, ansatz_reps)))

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean)
        return z_mean + eps * torch.exp(z_log_var / 2.)

    def encode_mean(self, x):
        """Deterministic latent (the posterior mean) for reconstruction/interpolation."""
        return self.z_mean(self.encoder(x))

    def decode(self, z):
        angles = self.preprocessor(z)  # (B, n_qubit) real rotation angles
        return decoder_distribution(angles, self.decoder_theta,
                                    self.n_qubit, self.fm_reps, self.ansatz_reps)

    def forward(self, x):
        h = self.encoder(x)
        z_mean, z_log_var = self.z_mean(h), self.z_log_var(h)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z), z_mean, z_log_var
