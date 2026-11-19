"""Stores one rollout (numStepsPerEnv steps from every parallel env) and computes GAE.
Also hands out shuffled minibatches for the PPO update.
"""

import torch


class RolloutBuffer:
    def __init__(self, numSteps, numEnvs, actorDim, criticDim, actDim, device):
        self.numSteps = numSteps
        self.numEnvs = numEnvs
        shape = (numSteps, numEnvs)

        def zeros(*extra):
            return torch.zeros(*shape, *extra, device=device)

        self.actorObs = zeros(actorDim)
        self.criticObs = zeros(criticDim)
        self.actions = zeros(actDim)
        self.logProbs = zeros()
        self.rewards = zeros()
        self.dones = zeros()
        self.values = zeros()
        self.advantages = zeros()
        self.returns = zeros()
        self.step = 0

    def add(self, actorObs, criticObs, actions, logProbs, rewards, dones, values):
        t = self.step
        self.actorObs[t] = actorObs
        self.criticObs[t] = criticObs
        self.actions[t] = actions
        self.logProbs[t] = logProbs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.step += 1

    @torch.no_grad()
    def computeReturns(self, lastValues, gamma, lam):
        # generalized advantage estimation, walking backwards through the rollout.
        # (1 - done) cuts the sum at episode boundaries
        gae = torch.zeros_like(lastValues)
        for t in reversed(range(self.numSteps)):
            nextValues = lastValues if t == self.numSteps - 1 else self.values[t + 1]
            notDone = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nextValues * notDone - self.values[t]
            gae = delta + gamma * lam * notDone * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def miniBatches(self, numMiniBatches):
        total = self.numSteps * self.numEnvs
        batchSize = total // numMiniBatches
        flat = {
            "actorObs": self.actorObs.flatten(0, 1),
            "criticObs": self.criticObs.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "logProbs": self.logProbs.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
        }
        order = torch.randperm(total, device=self.actorObs.device)
        for i in range(numMiniBatches):
            idx = order[i * batchSize : (i + 1) * batchSize]
            yield {k: v[idx] for k, v in flat.items()}

    def clear(self):
        self.step = 0
