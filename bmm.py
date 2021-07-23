import torch
from torch import distributions as D

# Work in progress.


class BetaMixtureModel(torch.nn.Module):
    def __init__(self, num_components):
        super().__init__()

        self.num_components = num_components
        self.weights = torch.nn.Parameter(torch.rand(num_components), requires_grad=False)
        self.alphas = torch.nn.Parameter(torch.rand(num_components) * 2, requires_grad=False)
        self.betas = torch.nn.Parameter(torch.rand(num_components) * 2, requires_grad=False)

    def log_likelihood(self, x, epsilon=1e-4):
        x[x > 1 - epsilon] = 1 - epsilon
        x[x < epsilon] = epsilon
        l = [D.Beta(a, b).log_prob(x) * w for w, a, b in zip(self.weights, self.alphas, self.betas)]
        return torch.stack(l)

    def log_posterior(self, x):
        ll = self.log_likelihood(x)
        return ll - ll.logsumexp(dim=0)

    def posterior(self, x):
        return self.log_posterior(x).exp()

    def predict(self, x):
        return self.log_posterior(x).argmax(dim=0)

    def fit(self, x, epsilon=1e-4, num_steps=10):
        x = x.clone().detach()
        x[x > 1 - epsilon] = 1 - epsilon
        x[x < epsilon] = epsilon

        for _ in range(num_steps):
            self._em(x)

        return self

    def _em(self, x):
        g = self.posterior(x)
        x_bar = (x * g).sum(dim=1, keepdim=True) / g.sum(dim=1, keepdim=True)
        s2 = (g * (x - x_bar) ** 2).sum(dim=1) / g.sum(dim=1)
        x_bar = x_bar[:, 0]

        alphas = x_bar * (x_bar * (1 - x_bar) / s2 - 1)
        betas = alphas * (1 - x_bar) / x_bar
        weights = g.sum(dim=1) / x.numel()

        self.alphas.copy_(alphas)
        self.betas.copy_(betas)
        self.weights.copy_(weights)
