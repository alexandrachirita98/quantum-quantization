import numpy as np
import torch
import torch.nn as nn


def embed_gate(gate, qubit, n_qubits):
    """Embed a (2, 2) gate acting on `qubit` into the full 2^n x 2^n space via Kronecker products."""
    eye = torch.eye(2, dtype=torch.complex128, device=gate.device)
    mats = [eye] * n_qubits
    mats[qubit] = gate
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out

def rz_gate(theta):
    t = theta.to(torch.complex128)
    zero = torch.zeros((), dtype=torch.complex128, device=theta.device)
    return torch.stack([torch.stack([torch.exp(-0.5j * t), zero]),
                        torch.stack([zero, torch.exp(0.5j * t)])])

def ry_gate(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)

def rx_gate(theta):
    c = torch.cos(theta / 2).to(torch.complex128)
    s = torch.sin(theta / 2).to(torch.complex128)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])

def crx_unitary(theta, control, target, n_qubits):
    """Controlled-RX: apply RX(theta) to `target` when `control` is |1>.
    Built as P0_c (x) I  +  P1_c (x) RX_target, with P0 = |0><0|, P1 = |1><1|."""
    dev = theta.device
    p0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128, device=dev)
    p1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex128, device=dev)
    return embed_gate(p0, control, n_qubits) + \
        embed_gate(p1, control, n_qubits) @ embed_gate(rx_gate(theta), target, n_qubits)

def apply_unitary(U, state):
    """state: (B, 2^n) complex statevectors -> (B, 2^n), i.e. U @ psi per sample."""
    return torch.einsum('ij,bj->bi', U, state)

def amplitude_encode(vec, n_qubits):
    """TorchQuantum-style AmplitudeEncoder: zero-pad `vec` up to 2^n, L2-normalize, cast to complex.
    A latent of dim 2^(n - bottleneck) is thus padded into the first amplitudes of the n-qubit register."""
    b, d = vec.shape
    dim = 2 ** n_qubits
    if d < dim:
        vec = torch.cat([vec, torch.zeros(b, dim - d, dtype=vec.dtype, device=vec.device)], dim=1)
    vec = vec / vec.norm(dim=1, keepdim=True)
    return vec.to(torch.complex128)


def marginalize_bottleneck(state, n_qubits, bottleneck_qbits):
    """state: (B, 2^n) -> (B, 2^(n - bottleneck_qbits)) real latent amplitudes."""
    latent_dim = 2 ** (n_qubits - bottleneck_qbits)
    probs = state.abs() ** 2
    probs = probs.reshape(-1, latent_dim, 2 ** bottleneck_qbits).sum(dim=-1)
    return probs.sqrt()


def build_vmf_cdf(kappa, dim, grid=200000):
    """Inverse-CDF table for the vMF cosine marginal p(w) ~ exp(kappa*w) * (1 - w^2)^((dim-3)/2)."""
    eps = 1e-7
    x = torch.linspace(-1 + eps, 1 - eps, grid, dtype=torch.float64)
    y = kappa * x + torch.log(1 - x ** 2) * (dim - 3) / 2
    cdf = torch.cumsum(torch.exp(y - y.max()), dim=0)
    return x, cdf / cdf[-1].clone()

def sample_vmf(mu, vmf_x, vmf_cdf):
    """Draw z ~ vMF(mu, kappa) via the tangent-normal decomposition (mu: (B, d) unit vectors)."""
    u = torch.rand(mu.size(0), dtype=torch.float64)
    w = np.interp(u.numpy(), vmf_cdf.cpu().numpy(), vmf_x.cpu().numpy())
    w = torch.tensor(w, dtype=mu.dtype, device=mu.device).unsqueeze(1)
    eps = torch.randn_like(mu)
    nu = eps - (eps * mu).sum(dim=1, keepdim=True) * mu       # component orthogonal to mu
    nu = nn.functional.normalize(nu, p=2, dim=-1)
    return w * mu + (1 - w ** 2).clamp(min=0).sqrt() * nu


class DrugQAE(nn.Module):
    """Statevector port of models/model.py::DrugQAE. `n_qbits` total qubits; `bottleneck_qbits`
    are marginalized after the encoder to form the spherical latent of dim 2^(n_qbits - bottleneck_qbits)."""
    def __init__(self, n_qbits=6, n_blocks=3, bottleneck_qbits=2, kappa=20.0):
        super().__init__()
        self.n_qbits = n_qbits
        self.n_blocks = n_blocks
        self.bottleneck_qbits = bottleneck_qbits
        self.latent_dim = 2 ** (n_qbits - bottleneck_qbits)
        self.kappa = kappa

        # per-qubit RZ-RY-RZ angles for n_blocks+1 blocks, and CRX angles for the n_blocks entangling rings
        rot = lambda: nn.Parameter(0.1 * torch.randn(n_blocks + 1, n_qbits))
        ent = lambda: nn.Parameter(0.1 * torch.randn(n_blocks, n_qbits))
        self.enc_rz1, self.enc_ry, self.enc_rz2, self.enc_crx = rot(), rot(), rot(), ent()
        self.dec_rz1, self.dec_ry, self.dec_rz2, self.dec_crx = rot(), rot(), rot(), ent()

        vmf_x, vmf_cdf = build_vmf_cdf(kappa, self.latent_dim)
        self.register_buffer("vmf_x", vmf_x)
        self.register_buffer("vmf_cdf", vmf_cdf)

    def encoder_unitary(self):
        n = self.n_qbits
        U = torch.eye(2 ** n, dtype=torch.complex128, device=self.enc_ry.device)
        for k in range(self.n_blocks + 1):
            for i in range(n):
                U = embed_gate(rz_gate(self.enc_rz1[k, i]), i, n) @ U
                U = embed_gate(ry_gate(self.enc_ry[k, i]), i, n) @ U
                U = embed_gate(rz_gate(self.enc_rz2[k, i]), i, n) @ U
            if k != self.n_blocks:
                for i in range(n):
                    U = crx_unitary(self.enc_crx[k, i], i, (i + 1) % n, n) @ U
        return U

    def decoder_unitary(self):
        n = self.n_qbits
        U = torch.eye(2 ** n, dtype=torch.complex128, device=self.dec_ry.device)
        for k in range(self.n_blocks, -1, -1):
            if k != self.n_blocks:
                for i in range(n - 1, -1, -1):
                    U = crx_unitary(self.dec_crx[k, i], i, (i + 1) % n, n) @ U
            for i in range(n):
                U = embed_gate(rz_gate(self.dec_rz2[k, i]), i, n) @ U
                U = embed_gate(ry_gate(self.dec_ry[k, i]), i, n) @ U
                U = embed_gate(rz_gate(self.dec_rz1[k, i]), i, n) @ U
        return U

    def encode(self, x):
        """x: (B, 2^n) input amplitudes -> (B, latent_dim) unit-vector latent means mu."""
        state = apply_unitary(self.encoder_unitary(), amplitude_encode(x, self.n_qbits))
        return marginalize_bottleneck(state, self.n_qbits, self.bottleneck_qbits)

    def decode(self, z):
        """z: (B, latent_dim) latent unit vectors -> (B, 2^n) reconstructed amplitudes (>= 0)."""
        state = apply_unitary(self.decoder_unitary(), amplitude_encode(z, self.n_qbits))
        return state.abs()

    def forward(self, x):
        mu = self.encode(x)
        z = sample_vmf(mu, self.vmf_x, self.vmf_cdf)
        return mu, self.decode(z)
