"""Actor and critic networks, plus a running mean/std normalizer for observations.
The actor is a Gaussian policy: an MLP outputs the mean and a learned log std sets the noise.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningNorm(nn.Module):
    # keeps a running mean/var of everything it has seen. stored as buffers so they get
    # saved in the checkpoint and the policy sees the same scaling at test time
    def __init__(self, dim, eps=1e-8, clip=5.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x):
        batchMean = x.mean(dim=0)
        batchVar = x.var(dim=0, unbiased=False)
        batchCount = x.shape[0]

        delta = batchMean - self.mean
        total = self.count + batchCount
        self.mean += delta * batchCount / total
        m2 = self.var * self.count + batchVar * batchCount + delta**2 * self.count * batchCount / total
        self.var = m2 / total
        self.count = total

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


def mlp(inDim, hiddenDims, outDim):
    layers = []
    for h in hiddenDims:
        layers += [nn.Linear(inDim, h), nn.ELU()]
        inDim = h
    layers.append(nn.Linear(inDim, outDim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obsDim, actDim, hiddenDims=(512, 256, 128), initStd=1.0):
        super().__init__()
        self.norm = RunningNorm(obsDim)
        self.net = mlp(obsDim, hiddenDims, actDim)
        self.logStd = nn.Parameter(torch.full((actDim,), float(initStd)).log())

    def dist(self, obs):
        mean = self.net(self.norm(obs))
        return Normal(mean, self.logStd.exp().expand_as(mean))

    def forward(self, obs):
        # deterministic action (the mean), used for playback and evaluation
        return self.net(self.norm(obs))


class Critic(nn.Module):
    def __init__(self, obsDim, hiddenDims=(512, 256, 128)):
        super().__init__()
        self.norm = RunningNorm(obsDim)
        self.net = mlp(obsDim, hiddenDims, 1)

    def forward(self, obs):
        return self.net(self.norm(obs)).squeeze(-1)
