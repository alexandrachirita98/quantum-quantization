import torch
import torch.nn as nn


class Poisson:
    def __init__(self, log_rate, temp=0.0, n_exp=64, eps=1e-6):
        self.log_rate = log_rate
        self.rate = torch.exp(log_rate) + eps
        self.temp = temp
        self.n_exp = n_exp
        self._exp = torch.distributions.Exponential(self.rate)

    def rsample(self):
        if self.temp == 0.0:
            return self.sample()
        x = self._exp.rsample((self.n_exp,))
        times = torch.cumsum(x, dim=0)
        logits = (1 - times) / self.temp
        indicator = torch.sigmoid(logits)
        z = indicator.sum(0)
        return z

    def sample(self):
        return torch.poisson(self.rate)

    @property
    def mean(self):
        return self.rate


class PoissonVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, n_latents=64, n_exp=64):
        super().__init__()
        self.n_latents = n_latents
        self.n_exp = n_exp

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_latents),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )
        # learned prior log-rate, shared across the batch
        self.log_rate = nn.Parameter(torch.zeros(n_latents))

    def infer(self, x, temp=1.0):
        log_dr = self.encoder(x.view(x.size(0), -1))
        log_r = self.log_rate.expand(len(log_dr), -1) + log_dr
        dist = Poisson(log_r, temp=temp, n_exp=self.n_exp)
        return dist, log_dr

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x, temp=1.0):
        dist, log_dr = self.infer(x, temp=temp)
        spks = dist.rsample()
        y = self.decode(spks)
        return y, spks, log_dr

    def loss_kl(self, log_dr):
        log_r = self.log_rate.expand(len(log_dr), -1)
        f = 1 + torch.exp(log_dr) * (log_dr - 1)
        return torch.exp(log_r) * f

    @torch.no_grad()
    def sample(self, n, temp=0.0):
        log_r = self.log_rate.expand(n, -1)
        dist = Poisson(log_r, temp=temp, n_exp=self.n_exp)
        spks = dist.rsample()
        return self.decode(spks), spks
