import torch
import torch.nn as nn


def to_distributions(images, res=8):
    x = images.reshape(-1, 1, 28, 28).double() / 255.
    x = nn.functional.adaptive_avg_pool2d(x, (res, res)).reshape(-1, res * res)
    x = x + 1e-6  # avoid an exact all-zero (unnormalizable) distribution
    return x / x.sum(dim=1, keepdim=True)  # L1-normalize -> distribution over 2^n bitstrings


n_qubit = 6
n_states = 2 ** n_qubit

# bit_table[i, b] = i-th bit (qubit i, MSB-first) of basis index b; used to read off per-qubit marginals
bit_table = torch.tensor(
    [[(b >> (n_qubit - 1 - i)) & 1 for b in range(n_states)] for i in range(n_qubit)],
    dtype=torch.float64)

def to_marginals(dist):
    return dist @ bit_table.t()  # (B, n_qubit): P(qubit i = 1), the encoder's measurement input


def reconstruction_loss(recon, target):
    smoothed = (recon + 2 ** -18) / (1 + 2 ** -18 * recon.size(-1))
    return -(target * torch.log(smoothed)).sum(-1).mean()

def kl_divergence(z_mean, z_log_var):
    kl = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=-1)
    return kl.mean()


def dist_to_image(prob, res=8):
    prob = prob.real.clamp(min=0)
    prob = prob / prob.sum(dim=-1, keepdim=True)
    return prob.reshape(-1, res, res)
