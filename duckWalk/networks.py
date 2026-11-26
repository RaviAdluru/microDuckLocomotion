import torch
import torch.nn as nn
from torch.distributions import Normal

def mlp(inDim, hiddenDim, outDim):
    # takes an arbitrary number of hidden layers for the 
    # NN and outputs each size from input -> output dimensions
    layers = [];
    prev = inDim
    for i in hiddenDim:
        layers.append(nn.Linear(prev, i))
        layers.append(nn.ELU())
        prev = i
    layers.append(nn.Linear(prev, outDim))
    return nn.Sequential(*layers)

class RunningNorm(nn.Module):
    # rescales every input feature to mean of about 0, std of about 11 using, updating across
    # training steps
    def __init__(self, dim, eps=1e-8, clip=5.0):
        super().__init__()  
        self.eps = eps
        self.clip = clip # clips prevents edge cases like falling and whatnot from falling out
        # of the standard deviation amount be an unreasonable amount

        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))  # readings seen so far

    @torch.no_grad()
    def update(self, x):
        # one for each sim training step a running mean and variance is updated without
        # storing old means and variances
        batchMean = x.mean(dim=0)
        batchVar = x.var(dim=0, unbiased=False)
        batchCount = x.shape[0]

        delta = batchMean - self.mean # diff between sim step mean vs running mean
        total = self.count + batchCount # N = n + m

        # variance first updated first because it needs old mean
        m2 = self.var * self.count + batchVar * batchCount + delta**2 * self.count * batchCount / total
        self.var = m2 / total
        self.mean += delta * batchCount / total # mean proportional update
        self.count = total

    def forward(self, x):
        # z = (x - mean) / std, ensures high z scores dont cause crazy variance
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class Critic(nn.Module):
    # predicts a number for each sample from a few given features about the duck
    def __init__(self, simDims, hiddenDims): # simDims = known things about duck in the sim, but
                                             # not from the sensor itself
        super().__init__()
        self.norm = RunningNorm(simDims) # norms
        self.net = mlp(simDims, hiddenDims, 1) # makes mlp for simDims -> hiddenDims
                                                # predicts on number per state

    def forward(self, obs):
        z = self.norm(obs) # rescales for mean 0 std 1
        return self.net(z).squeeze(-1)

class Actor(nn.Module):
    # outputs bell curve for each output for exploration so training varies
    # outputs just mean when we want its best guess in play.py and evaluate.py for validation
    def __init__(self, obsDim, actDim, hiddenDims=(512, 256, 128), initStd=1.0):
        super().__init__()
        self.norm = RunningNorm(obsDim) # rescales for inputs
        self.net = mlp(obsDim, hiddenDims, actDim) # 51 -> 512 -> 256 -> 128 -> 14 means, one per joint

        # spread for the learning range for each joint (learns with log so exp can be applied and variance is positive for gradient descent to work)
        self.logStd = nn.Parameter(torch.full((actDim,), float(initStd)).log())

    def dist(self, obs):
        # distribution for entropy
        # trainer uses sample while ppo calls for an entropy term
        mean = self.net(self.norm(obs)) # targeted angle offset for each joint
        std = self.logStd.exp().expand_as(mean) # spreads copied for learning to all ducks
        return Normal(mean, std)

    def forward(self, obs):
        # best guess without randomness for use and validation
        return self.net(self.norm(obs))