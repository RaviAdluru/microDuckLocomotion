"""PPO update: clipped surrogate loss, clipped value loss and an entropy bonus.
Learning rate is adjusted each minibatch to keep the policy's KL change near targetKl.
"""

import torch
import torch.nn as nn


class PPO:
    def __init__(self, actor, critic, cfg):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.learningRate = cfg.learningRate
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=self.learningRate
        )

    def update(self, buffer):
        cfg = self.cfg
        stats = {"policyLoss": 0.0, "valueLoss": 0.0, "entropy": 0.0, "approxKl": 0.0, "clipFraction": 0.0}
        numUpdates = 0

        for _ in range(cfg.numEpochs):
            for batch in buffer.miniBatches(cfg.numMiniBatches):
                dist = self.actor.dist(batch["actorObs"])
                newLogProbs = dist.log_prob(batch["actions"]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # surrogate: L = E[min(r*A, clip(r, 1-eps, 1+eps)*A)],  r = pi_new / pi_old
                logRatio = newLogProbs - batch["logProbs"]
                ratio = logRatio.exp()
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - cfg.clipRatio, 1 + cfg.clipRatio) * adv
                policyLoss = -torch.min(surr1, surr2).mean()

                # value loss, also clipped so the critic can't jump too far in one update
                values = self.critic(batch["criticObs"])
                oldValues = batch["values"]
                valuesClipped = oldValues + (values - oldValues).clamp(-cfg.clipRatio, cfg.clipRatio)
                valueLoss = torch.max(
                    (values - batch["returns"]) ** 2, (valuesClipped - batch["returns"]) ** 2
                ).mean()

                loss = policyLoss + cfg.valueCoef * valueLoss - cfg.entropyCoef * entropy

                with torch.no_grad():
                    approxKl = ((ratio - 1) - logRatio).mean().item()
                    clipFraction = ((ratio - 1).abs() > cfg.clipRatio).float().mean().item()
                self.adaptLearningRate(approxKl)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), cfg.maxGradNorm
                )
                self.optimizer.step()

                stats["policyLoss"] += policyLoss.item()
                stats["valueLoss"] += valueLoss.item()
                stats["entropy"] += entropy.item()
                stats["approxKl"] += approxKl
                stats["clipFraction"] += clipFraction
                numUpdates += 1

        stats = {k: v / numUpdates for k, v in stats.items()}
        stats["learningRate"] = self.learningRate
        stats["actionStd"] = self.actor.logStd.exp().mean().item()

        # how much of the return variance the critic explains. 1 is perfect, <=0 is useless
        with torch.no_grad():
            values = buffer.values.flatten()
            returns = buffer.returns.flatten()
            stats["explainedVariance"] = (1 - (returns - values).var() / (returns.var() + 1e-8)).item()
        return stats

    def adaptLearningRate(self, kl):
        if kl > 2.0 * self.cfg.targetKl:
            self.learningRate = max(1e-5, self.learningRate / 1.5)
        elif kl < 0.5 * self.cfg.targetKl:
            self.learningRate = min(1e-2, self.learningRate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learningRate
