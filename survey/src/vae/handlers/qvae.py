import torch
import torch.nn as nn


def to_amplitude_states(images, res=8):
    x = images.reshape(-1, 1, 28, 28).double() / 255.
    x = nn.functional.adaptive_avg_pool2d(x, (res, res)).reshape(-1, res * res)
    x = x + 1e-6  # avoid an exact all-zero (unnormalizable) state
    x = x / x.norm(dim=1, keepdim=True)
    return x


def to_density_matrices(x):
    return torch.einsum('bi,bj->bij', x, x.conj()).to(torch.complex128)


def state_fidelity(rho, sigma):
    """Uhlmann fidelity, batched over a leading (...) dimension: F = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    torch.linalg.eigh/eigvalsh operate on the trailing two dims and are already batched, so
    this needs no Python loop over the batch.
    """
    evals, evecs = torch.linalg.eigh(rho)
    evals = evals.clamp(min=0).to(torch.complex128)
    sqrt_rho = evecs @ torch.diag_embed(evals.sqrt()) @ evecs.conj().transpose(-2, -1)
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner = (inner + inner.conj().transpose(-2, -1)) / 2  # enforce Hermitian
    inner_evals = torch.linalg.eigvalsh(inner).clamp(min=0)
    return inner_evals.sqrt().sum(-1) ** 2

def von_neumann_entropy(rho):
    evals = torch.linalg.eigvalsh(rho).clamp(min=1e-12)
    return -(evals * evals.log2()).sum(-1)

def reconstruction_loss(recon, target):
    return -state_fidelity(recon, target).real.mean()

def jsd_regularizer(latent):
    dim = latent.size(-1)
    max_mixed = torch.eye(dim, dtype=torch.complex128, device=latent.device) / dim
    max_mixed_b = max_mixed.expand_as(latent)
    m = 0.5 * (latent + max_mixed_b)
    ent_latent = von_neumann_entropy(latent)
    ent_mixed = von_neumann_entropy(max_mixed)
    ent_m = von_neumann_entropy(m)
    # JSD(a||b) = H(0.5a+0.5b) - 0.5*H(a) - 0.5*H(b), matching core.py's sign convention (minimized)
    jsd = ent_m - 0.5 * ent_latent - 0.5 * ent_mixed
    return jsd.mean()


def random_pure_states(n, dim):
    real = torch.randn(n, dim, dtype=torch.float64)
    imag = torch.randn(n, dim, dtype=torch.float64)
    v = (real + 1j * imag).to(torch.complex128)
    v = v / v.norm(dim=1, keepdim=True)
    return v

def density_diag_to_image(rho_batch, res=8):
    probs = rho_batch.diagonal(dim1=-2, dim2=-1).real.clamp(min=0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs.reshape(-1, res, res)
