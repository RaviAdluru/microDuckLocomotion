"""Test checkpoints on the eval terrains, including ones never seen in training.
Each env gets a fixed forward speed command; we record falls, velocity tracking error and distance.
    uv run scripts/evaluate.py runs/a/model_3000.pt --split validation --out results/val.csv
    uv run scripts/evaluate.py runs/a/model_3000.pt --split test --out results/test.csv
The combined courses need a longer episode so ducks cross tile seams:
    uv run scripts/evaluate.py runs/a/model_3000.pt --terrains combinedSeen combinedAll --seconds 20
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from duckWalk.envConfig import makeEnvCfg
from duckWalk.terrains import HELD_OUT, SPLITS, TIERS, TRAINING_TYPE, evalTerrain
from duckWalk.trainer import loadActor, loadCheckpointInfo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def buildEvalEnv(trainCfg, terrainName, difficulty, args):
    cfg = makeEnvCfg(
        rewardSet=trainCfg["rewardSet"],
        terrainSet=trainCfg["terrainSet"],
        historyLength=trainCfg["historyLength"],
        domainRand=args.domainRand,
        numEnvs=args.numEnvs,
        terrainOverride=evalTerrain(terrainName, difficulty, seed=args.seed),
    )
    cfg.episode_length_s = args.seconds
    if not args.pushes:
        cfg.events.pop("push_robot", None)

    # straight-ahead commands only, spread evenly over [minSpeed, maxSpeed], never resampled
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (args.minSpeed, args.maxSpeed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.ranges.heading = None
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    # Pollen's command class sends 15% of envs a turn-in-place command (zero forward speed).
    # with turning zeroed above, those ducks would just stand still. every eval duck should walk
    twist.rel_turn_in_place_envs = 0.0
    twist.resampling_time_range = (args.seconds + 1.0, args.seconds + 1.0)

    env = ManagerBasedRlEnv(cfg, device=args.device)
    env.seed(args.seed)
    return env


def trainedUpTo(terrainName, reach):
    # highest difficulty the policy trained on for this terrain's type, or None if it doesn't
    # apply (novel/combined terrains, or a checkpoint trained without a curriculum)
    if not reach or terrainName not in TRAINING_TYPE:
        return None
    level = reach["levels"].get(TRAINING_TYPE[terrainName])
    if level is None:
        return None
    # row k covers difficulties k/numRows to (k+1)/numRows
    return (level + 1) / reach["numRows"]


@torch.no_grad()
def runEpisode(env, actor):
    obs, _ = env.reset()
    robot = env.scene["robot"]
    n = env.num_envs
    dt = env.step_dt
    numSteps = int(env.max_episode_length)

    command = env.command_manager.get_command("twist")[:, 0].clone()
    active = torch.ones(n, dtype=torch.bool, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    velError = torch.zeros(n, device=env.device)
    distance = torch.zeros(n, device=env.device)
    stepsAlive = torch.zeros(n, device=env.device)

    for _ in range(numSteps):
        obs, _, terminated, truncated, _ = env.step(actor(obs["actor"]))
        vel = robot.data.root_link_lin_vel_b[:, :2]
        cmd = env.command_manager.get_command("twist")[:, :2]
        a = active.float()
        velError += (cmd - vel).norm(dim=1) * a
        distance += vel[:, 0] * dt * a
        stepsAlive += a
        fell |= terminated & active
        active &= ~(terminated | truncated)
        if not active.any():
            break

    return {
        "command": command.cpu(),
        "fell": fell.cpu(),
        "velError": (velError / stepsAlive.clamp(min=1)).cpu(),
        "distance": distance.cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--split", choices=list(SPLITS), default="all", help="which terrain group to run")
    parser.add_argument("--terrains", nargs="+", default=None, help="overrides --split")
    parser.add_argument("--difficulties", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--minSpeed", type=float, default=0.1)
    parser.add_argument("--maxSpeed", type=float, default=0.4)
    parser.add_argument("--numEnvs", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1000, help="keep this different from training seeds")
    parser.add_argument("--pushes", action="store_true", help="keep random pushes on during eval")
    parser.add_argument("--domainRand", action="store_true", help="randomize mass/friction etc. during eval")
    parser.add_argument("--out", default="results/eval.csv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    terrains = args.terrains or SPLITS[args.split]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for ckptPath in args.checkpoints:
        actor, trainCfg = loadActor(ckptPath, args.device)
        iteration, reach = loadCheckpointInfo(ckptPath)
        for terrainName in terrains:
            for difficulty in args.difficulties:
                # same seed for every terrain/checkpoint, so every policy sees identical ground
                seedEverything(args.seed)
                env = buildEvalEnv(trainCfg, terrainName, difficulty, args)
                result = runEpisode(env, actor)
                env.close()

                upTo = trainedUpTo(terrainName, reach)
                # a "seen" terrain only counts as seen if training got to this difficulty
                withinTraining = "" if upTo is None else difficulty <= upTo + 1e-9
                if TIERS[terrainName] == "seen" and withinTraining is False:
                    print(
                        f"  note: {terrainName} at d={difficulty:.2f} is above what training reached "
                        f"(about {upTo:.1f}), so treat it as extrapolation, not seen"
                    )

                for i in range(len(result["fell"])):
                    rows.append(
                        {
                            "checkpoint": ckptPath,
                            "runName": trainCfg["runName"],
                            "iteration": iteration,
                            "trainSeed": trainCfg["seed"],
                            "evalSeed": args.seed,
                            "terrain": terrainName,
                            "heldOut": terrainName in HELD_OUT,
                            "tier": TIERS[terrainName],
                            "difficulty": difficulty,
                            "trainedUpTo": "" if upTo is None else round(upTo, 3),
                            "withinTraining": withinTraining,
                            "pushes": args.pushes,
                            "domainRand": args.domainRand,
                            "commandSpeed": round(result["command"][i].item(), 4),
                            "fell": int(result["fell"][i]),
                            "velError": round(result["velError"][i].item(), 4),
                            "distance": round(result["distance"][i].item(), 4),
                        }
                    )
                fell = result["fell"].float()
                fallRate = fell.mean().item()
                # standard error of a fall rate from n independent ducks: sqrt(p(1-p)/n)
                fallSe = (fallRate * (1 - fallRate) / len(fell)) ** 0.5
                velErr = result["velError"].mean().item()
                print(
                    f"{trainCfg['runName']:>16}  it {iteration:5d}  {terrainName:>16}  "
                    f"{TIERS[terrainName]:>8}  d={difficulty:.2f}  "
                    f"fall {fallRate:6.1%} +/- {fallSe:.1%}  velErr {velErr:.3f} m/s"
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
