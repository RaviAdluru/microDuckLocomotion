"""The training loop: collect a rollout from all envs, run a PPO update, log, checkpoint.
Works with any mjlab env that has "actor" and "critic" observation groups (we also use it on cartpole).
"""

import os
import time
from collections import deque
from dataclasses import asdict

import torch

from duckWalk.networks import Actor, Critic
from duckWalk.ppo import PPO
from duckWalk.rolloutBuffer import RolloutBuffer


def buildModels(cfg, actorDim, criticDim, actDim, device):
    actor = Actor(actorDim, actDim, cfg.hiddenDims, cfg.initStd).to(device)
    critic = Critic(criticDim, cfg.hiddenDims).to(device)
    return actor, critic


def saveCheckpoint(path, actor, critic, ppo, iteration, cfg, dims):
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": ppo.optimizer.state_dict(),
            "learningRate": ppo.learningRate,
            "iteration": iteration,
            "config": asdict(cfg),
            "dims": dims,
        },
        path,
    )


def loadActor(path, device):
    # returns the trained actor plus the config it was trained with, so play/eval can
    # rebuild the env with the same observation layout
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    dims = ckpt["dims"]
    actor = Actor(dims["actor"], dims["actions"], tuple(cfg["hiddenDims"]), cfg["initStd"]).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, cfg


def train(env, cfg, logger, runDir):
    device = env.device
    obs, _ = env.reset()
    actorDim = obs["actor"].shape[1]
    criticDim = obs["critic"].shape[1]
    actDim = env.action_manager.total_action_dim
    numEnvs = env.num_envs
    dims = {"actor": actorDim, "critic": criticDim, "actions": actDim}
    print(f"actor obs {actorDim}, critic obs {criticDim}, actions {actDim}, envs {numEnvs}")

    actor, critic = buildModels(cfg, actorDim, criticDim, actDim, device)
    ppo = PPO(actor, critic, cfg)
    buffer = RolloutBuffer(cfg.numStepsPerEnv, numEnvs, actorDim, criticDim, actDim, device)

    startIteration = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        ppo.optimizer.load_state_dict(ckpt["optimizer"])
        ppo.learningRate = ckpt["learningRate"]
        startIteration = ckpt["iteration"] + 1
        print(f"resumed from {cfg.resume} at iteration {startIteration}")

    epReturn = torch.zeros(numEnvs, device=device)
    epLength = torch.zeros(numEnvs, device=device)
    # episodes last much longer than one rollout, so average over the last 200 that finished
    finishedReturns = deque(maxlen=200)
    finishedLengths = deque(maxlen=200)
    os.makedirs(runDir, exist_ok=True)

    for iteration in range(startIteration, cfg.maxIterations):
        tStart = time.time()
        envLogs = {}
        stepRewardSum = 0.0

        # ---- collect a rollout ----
        with torch.no_grad():
            for _ in range(cfg.numStepsPerEnv):
                actorObs, criticObs = obs["actor"], obs["critic"]
                actor.norm.update(actorObs)
                critic.norm.update(criticObs)

                dist = actor.dist(actorObs)
                actions = dist.sample()
                logProbs = dist.log_prob(actions).sum(-1)
                values = critic(criticObs)

                obs, rewards, terminated, truncated, extras = env.step(actions)
                dones = (terminated | truncated).float()
                epReturn += rewards
                epLength += 1
                stepRewardSum += rewards.mean().item()

                # an episode cut off by the time limit didn't actually end, so bootstrap
                # with the value estimate instead of treating it as a dead end
                trainRewards = rewards + cfg.gamma * values * truncated.float()
                buffer.add(actorObs, criticObs, actions, logProbs, trainRewards, dones, values)

                doneIds = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(doneIds) > 0:
                    finishedReturns.extend(epReturn[doneIds].tolist())
                    finishedLengths.extend(epLength[doneIds].tolist())
                    epReturn[doneIds] = 0
                    epLength[doneIds] = 0

                # mjlab reports per-term episode rewards, terrain levels etc. when envs reset
                for key, value in extras.get("log", {}).items():
                    envLogs.setdefault(key, []).append(float(value))

            lastValues = critic(obs["critic"])
            buffer.computeReturns(lastValues, cfg.gamma, cfg.lam)
        collectTime = time.time() - tStart

        # ---- learn ----
        tLearn = time.time()
        stats = ppo.update(buffer)
        buffer.clear()
        learnTime = time.time() - tLearn

        # ---- log ----
        stepsPerSec = cfg.numStepsPerEnv * numEnvs / (collectTime + learnTime)
        logValues = {f"train/{k}": v for k, v in stats.items()}
        logValues["perf/stepsPerSec"] = stepsPerSec
        logValues["perf/collectTime"] = collectTime
        logValues["perf/learnTime"] = learnTime
        logValues["rollout/meanStepReward"] = stepRewardSum / cfg.numStepsPerEnv
        if finishedReturns:
            logValues["episode/return"] = sum(finishedReturns) / len(finishedReturns)
            logValues["episode/length"] = sum(finishedLengths) / len(finishedLengths)
        for key, vals in envLogs.items():
            logValues[f"env/{key}"] = sum(vals) / len(vals)
        terrain = getattr(env.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None)
        if levels is not None:
            logValues["terrain/meanLevel"] = levels.float().mean().item()
            logValues["terrain/maxLevel"] = levels.max().item()
        logger.log(logValues, iteration)

        if iteration % 10 == 0:
            ret = logValues.get("episode/return", float("nan"))
            length = logValues.get("episode/length", float("nan"))
            print(
                f"iter {iteration:5d}  return {ret:8.2f}  epLen {length:6.1f}  "
                f"kl {stats['approxKl']:.4f}  lr {stats['learningRate']:.1e}  {stepsPerSec:,.0f} steps/s"
            )

        if iteration % cfg.saveInterval == 0 or iteration == cfg.maxIterations - 1:
            path = os.path.join(runDir, f"model_{iteration}.pt")
            saveCheckpoint(path, actor, critic, ppo, iteration, cfg, dims)

    return actor, critic
