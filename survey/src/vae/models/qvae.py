import torch
import torch.nn as nn


def embed_gate(gate, qubit, n_qubits):
    """gate: (2, 2) complex tensor already on the target device."""
    eye = torch.eye(2, dtype=torch.complex128, device=gate.device)
    mats = [eye] * n_qubits
    mats[qubit] = gate
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out

def ry_gate(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)

def rzz_unitary(theta, q1, q2, n_qubits):
    device = theta.device
    z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)
    zz = embed_gate(z, q1, n_qubits) @ embed_gate(z, q2, n_qubits)
    dim = 2 ** n_qubits
    eye = torch.eye(dim, dtype=torch.complex128, device=device)
    # exp(-i*theta/2*ZZ) with ZZ diagonal +-1 -> cos(theta/2)*I - i*sin(theta/2)*ZZ
    return torch.cos(theta / 2) * eye - 1j * torch.sin(theta / 2) * zz

def ising_interaction_unitary(theta, n_layers, n_qubits):
    """theta: 1D real tensor of length n_layers * (n_qubits*(n_qubits-1)/2 + n_qubits)."""
    dim = 2 ** n_qubits
    U = torch.eye(dim, dtype=torch.complex128, device=theta.device)
    idx = 0
    for _ in range(n_layers):
        for q1 in range(n_qubits):
            for q2 in range(q1 + 1, n_qubits):
                U = rzz_unitary(theta[idx], q1, q2, n_qubits) @ U
                idx += 1
        for q in range(n_qubits):
            U = embed_gate(ry_gate(theta[idx]).to(theta.device), q, n_qubits) @ U
            idx += 1
    return U

def num_ising_params(n_layers, n_qubits):
    return n_layers * (n_qubits * (n_qubits - 1) // 2 + n_qubits)


def batched_partial_trace_keep(rho, keep, n_qubits):
    """Trace out all qubits of a batch of n_qubits density matrices except `keep`.

    rho: (B, 2^n_qubits, 2^n_qubits). Vectorized over the batch dimension B (no Python loop
    over samples) since B is the axis this notebook's batch size actually scales.
    """
    trace_out = [q for q in range(n_qubits) if q not in keep]
    b = rho.size(0)
    dims = [2] * n_qubits
    rho_t = rho.reshape([b] + dims + dims)
    # ket axes are 1..n_qubits, bra axes are n_qubits+1..2*n_qubits (axis 0 is the batch dim)
    for q in sorted(trace_out, reverse=True):
        ket_axis = 1 + q
        bra_axis = 1 + q + n_qubits
        rho_t = torch.diagonal(rho_t, dim1=ket_axis, dim2=bra_axis, offset=0).sum(-1)
        # torch.diagonal moves the summed axis to the end; rebuild bookkeeping
        n_qubits -= 1
        dims = dims[:q] + dims[q + 1:]
        rho_t = rho_t.reshape([b] + dims + dims)
    dim = 2 ** len(keep)
    rho_out = rho_t.reshape(b, dim, dim)
    trace = rho_out.diagonal(dim1=-2, dim2=-1).sum(-1).real
    return rho_out / trace.view(-1, 1, 1)


class QVAE(nn.Module):
    def __init__(self, n_qubit=6, n_trash=2, n_layers=3):
        super().__init__()
        self.n_qubit = n_qubit
        self.n_trash = n_trash
        self.n_latent = n_qubit - n_trash
        self.n_layers = n_layers

        n_params = num_ising_params(n_layers, n_qubit)
        self.encoder_theta = nn.Parameter(0.1 * torch.randn(n_params))
        self.decoder_theta = nn.Parameter(0.1 * torch.randn(n_params))
        self.n_latent_dim = 2 ** self.n_latent

        zero = torch.zeros(2 ** n_trash, 2 ** n_trash, dtype=torch.complex128)
        zero[0, 0] = 1.0
        self.register_buffer("reconstruction_register", zero)

    def encode(self, rho):
        """rho: (B, 2^n_qubit, 2^n_qubit) input density matrices -> (B, 2^n_latent, ...) latents."""
        U = ising_interaction_unitary(self.encoder_theta, self.n_layers, self.n_qubit)
        evolved = U @ rho @ U.conj().T
        latent_qubits = list(range(self.n_trash, self.n_qubit))  # trace out the first n_trash
        return batched_partial_trace_keep(evolved, latent_qubits, self.n_qubit)

    def decode(self, latent):
        """latent: (B, 2^n_latent, ...) -> (B, 2^n_qubit, ...) reconstructed density matrices."""
        b = latent.size(0)
        reg = self.reconstruction_register
        # batched kron(reg, latent[i]) via an outer-product-of-blocks reshape, avoiding a
        # Python loop over the batch dimension
        full = reg[None, :, None, :, None] * latent[:, None, :, None, :]
        full = full.reshape(b, reg.size(0) * self.n_latent_dim, reg.size(0) * self.n_latent_dim)
        U = ising_interaction_unitary(self.decoder_theta, self.n_layers, self.n_qubit)
        return U @ full @ U.conj().T

    def forward(self, rho):
        latent = self.encode(rho)
        recon = self.decode(latent)
        return recon, latent
