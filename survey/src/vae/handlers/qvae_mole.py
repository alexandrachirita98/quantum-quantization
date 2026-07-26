import torch
import torch.nn as nn


def to_amplitude_states(images, res=8):
    x = images.reshape(-1, 1, 28, 28).double() / 255.
    x = nn.functional.adaptive_avg_pool2d(x, (res, res)).reshape(-1, res * res)
    x = x + 1e-6  # avoid an exact all-zero (unnormalizable) state
    x = x / x.norm(dim=1, keepdim=True)
    return x


def fidelity_loss(x, recon):
    overlap = torch.einsum('bi,bi->b', x, recon)
    return (1 - overlap ** 2).mean()


def amplitudes_to_image(recon, res=8):
    probs = recon ** 2
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs.reshape(-1, res, res)
