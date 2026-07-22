"""Reproduce Figures 2-5 from Kingma & Welling, "Auto-Encoding Variational
Bayes" (https://arxiv.org/pdf/1312.6114), using one of the baseline VAEs in
survey/src/related_work/vae/models/baseline/ (novicestone_vae.py or
pytorch_vae.py, selectable via the `model_name` constructor argument).

Notes on scope (see paper for the original figures):
- Fig. 2/3 in the paper compare AEVB against wake-sleep and Monte Carlo EM
  on MNIST and Frey Face. Only AEVB is implemented in this repo, and only
  MNIST is available locally, so plot_figure2/plot_figure3 reproduce the
  AEVB lower-bound learning curve only.
- Fig. 4 requires a 2-D latent space so two coordinates can be swept on a
  grid. plot_figure4 trains a dedicated VAE with latent_size=2 for this.
- Fig. 5 visualizes a higher-dimensional latent space via t-SNE.

pytorch_vae.VAE hardcodes its layer sizes (784-400-20-400-784) instead of
taking them as constructor arguments, so `_build_pytorch_vae` below replaces
the latent-facing layers after construction to support the varying Nz and
hidden sizes needed across figures 2, 4 and 5. novicestone_vae.VAE already
takes these sizes as constructor arguments.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.manifold import TSNE
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
try:
    get_ipython()  # noqa: F821 - defined by IPython/Jupyter kernels
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models", "baseline"))
import novicestone_vae  # noqa: E402
import pytorch_vae  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "data")

MODEL_BUILDERS = {}


def _build_novicestone_vae(latent_size, hidden_size=500):
    return novicestone_vae.VAE(
        input_size=784, hidden_size=hidden_size,
        latent_size=latent_size, data_type="binary",
    )


def _build_pytorch_vae(latent_size, hidden_size=400):
    """pytorch_vae.VAE is hardcoded to 784-400-20-400-784; rebuild the
    latent-facing layers so latent_size/hidden_size can vary per figure.
    """
    model = pytorch_vae.VAE()
    model.fc1 = nn.Linear(784, hidden_size)
    model.fc21 = nn.Linear(hidden_size, latent_size)
    model.fc22 = nn.Linear(hidden_size, latent_size)
    model.fc3 = nn.Linear(latent_size, hidden_size)
    model.fc4 = nn.Linear(hidden_size, 784)
    return model


MODEL_BUILDERS = {
    "novicestone": _build_novicestone_vae,
    "pytorch": _build_pytorch_vae,
}


def _forward(model_name, model, x):
    """Normalize each model's forward() output to (x_recon, mu, logvar)."""
    if model_name == "novicestone":
        mu, logvar, x_recon = model(x)
    else:
        x_recon, mu, logvar = model(x)
    return x_recon, mu, logvar


def vae_loss(x, x_recon, mu, logvar):
    bce = torch.nn.functional.binary_cross_entropy(
        x_recon, x, reduction="sum"
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


class VAEPaperFigureHandler:
    """Trains baseline VAE models and plots Figures 2-5 from the paper."""

    def __init__(self, model_name="pytorch", batch_size=128, device=None,
                 seed=1, data_dir=DATA_DIR):
        if model_name not in MODEL_BUILDERS:
            raise ValueError(
                f"Unknown model_name {model_name!r}; expected one of "
                f"{sorted(MODEL_BUILDERS)}"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.data_dir = data_dir
        torch.manual_seed(seed)

        self.train_loader = DataLoader(
            datasets.MNIST(
                self.data_dir, train=True, download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=batch_size, shuffle=True,
        )
        self.test_loader = DataLoader(
            datasets.MNIST(
                self.data_dir, train=False, download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=batch_size, shuffle=False,
        )

    def _train_vae(self, latent_size, epochs, hidden_size=None, lr=1e-3,
                    track_curve=False):
        builder = MODEL_BUILDERS[self.model_name]
        kwargs = {"hidden_size": hidden_size} if hidden_size is not None else {}
        model = builder(latent_size, **kwargs).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        samples_seen, neg_lower_bound = [], []
        total_samples = 0

        model.train()
        for epoch in range(1, epochs + 1):
            for x, _ in self.train_loader:
                x = x.view(-1, 784).to(self.device)
                optimizer.zero_grad()
                x_recon, mu, logvar = _forward(self.model_name, model, x)
                loss = vae_loss(x, x_recon, mu, logvar)
                loss.backward()
                optimizer.step()

                total_samples += x.size(0)
                if track_curve:
                    samples_seen.append(total_samples)
                    neg_lower_bound.append(loss.item() / x.size(0))

        return model, samples_seen, neg_lower_bound

    def _learning_curve(self, latent_size, epochs, hidden_size=None):
        _, samples_seen, neg_lower_bound = self._train_vae(
            latent_size, epochs, hidden_size=hidden_size, track_curve=True,
        )
        # Smooth with a moving average, similar in spirit to the paper's plots.
        window = max(1, len(neg_lower_bound) // 200)
        if window > 1:
            kernel = np.ones(window) / window
            neg_lower_bound = np.convolve(neg_lower_bound, kernel, mode="valid")
            samples_seen = samples_seen[window - 1:]
        return np.array(samples_seen), np.array(neg_lower_bound)

    def plot_figure2(self, latent_sizes=(3, 5, 10, 20, 200), epochs=5,
                      save_path=None):
        """AEVB lower-bound learning curves on MNIST for varying Nz.

        Reproduces only the AEVB curves from Fig. 2 (wake-sleep and Monte
        Carlo EM are not implemented in this repo).
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        for nz in latent_sizes:
            samples_seen, neg_lb = self._learning_curve(nz, epochs)
            ax.plot(samples_seen, neg_lb, label=f"AEVB, Nz={nz}")

        ax.set_xscale("log")
        ax.set_xlabel("Number of training samples evaluated")
        ax.set_ylabel("-Lower bound (per sample)")
        ax.set_title("Figure 2: MNIST, AEVB learning curves for varying Nz")
        ax.legend()
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    def plot_figure3(self, latent_size=10, epochs=5, save_path=None):
        """AEVB lower-bound learning curve on MNIST (single Nz), analogous
        to Fig. 3's comparison of optimization methods (AEVB only here).
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        samples_seen, neg_lb = self._learning_curve(latent_size, epochs)
        ax.plot(samples_seen, neg_lb, label=f"AEVB, Nz={latent_size}")

        ax.set_xscale("log")
        ax.set_xlabel("Number of training samples evaluated")
        ax.set_ylabel("-Lower bound (per sample)")
        ax.set_title("Figure 3: MNIST, AEVB optimization progress")
        ax.legend()
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    def plot_figure4(self, epochs=15, grid_size=20, image_shape=(28, 28),
                      save_path=None):
        """2-D latent manifold: decode a grid of latent codes obtained via
        the inverse CDF of the standard normal, spanning the unit square.
        """
        model, _, _ = self._train_vae(latent_size=2, epochs=epochs)
        model.eval()

        grid_x = norm.ppf(np.linspace(0.05, 0.95, grid_size))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, grid_size))

        h, w = image_shape
        canvas = np.zeros((h * grid_size, w * grid_size))
        with torch.no_grad():
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z = torch.tensor([[xi, yi]], dtype=torch.float32, device=self.device)
                    x_recon = model.decode(z).cpu().numpy().reshape(h, w)
                    canvas[i * h:(i + 1) * h, j * w:(j + 1) * w] = x_recon

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(canvas, cmap="gray_r")
        ax.set_title("Figure 4: Learned 2-D latent manifold (MNIST, Nz=2)")
        ax.axis("off")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    def plot_figure5(self, latent_size=20, epochs=10, n_samples=2000,
                      save_path=None):
        """t-SNE visualization of a higher-dimensional latent space,
        colored by digit label, analogous to Fig. 5.
        """
        model, _, _ = self._train_vae(latent_size=latent_size, epochs=epochs)
        model.eval()

        zs, labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.view(-1, 784).to(self.device)
                z_mean, _ = model.encode(x)
                zs.append(z_mean.cpu().numpy())
                labels.append(y.numpy())
                if sum(len(a) for a in zs) >= n_samples:
                    break

        z = np.concatenate(zs, axis=0)[:n_samples]
        labels = np.concatenate(labels, axis=0)[:n_samples]

        z_2d = TSNE(n_components=2, init="pca", random_state=0).fit_transform(z)

        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", s=8)
        ax.set_title(f"Figure 5: t-SNE of learned latent space (MNIST, Nz={latent_size})")
        legend = ax.legend(*scatter.legend_elements(), title="Digit", loc="best")
        ax.add_artist(legend)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    @staticmethod
    def _save_or_return(fig, save_path):
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            return save_path
        return fig


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "notebooks", "figures")
    os.makedirs(out_dir, exist_ok=True)

    handler = VAEPaperFigureHandler()
    handler.plot_figure2(save_path=os.path.join(out_dir, "figure2.png"))
    handler.plot_figure3(save_path=os.path.join(out_dir, "figure3.png"))
    handler.plot_figure4(save_path=os.path.join(out_dir, "figure4.png"))
    handler.plot_figure5(save_path=os.path.join(out_dir, "figure5.png"))
