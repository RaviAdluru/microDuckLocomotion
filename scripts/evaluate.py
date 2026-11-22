"""Test checkpoints on every eval terrain, including ones never seen in training.
Each env gets a fixed forward speed command; we record falls, velocity tracking error and distance.
    uv run scripts/evaluate.py runs/a/model_3000.pt runs/b/model_3000.pt --out results/eval.csv
"""

import argparse
import csv
import os

import torch
from mjlab.envs import ManagerBasedRlEnv

from duckWalk.envConfig import makeEnvCfg
from duckWalk.terrains import EVAL_TERRAINS, HELD_OUT, TIERS, evalTerrain
from duckWalk.trainer import loadActor


def buildEvalEnv(trainCfg, terrainName, difficulty, args):
    cfg = makeEnvCfg(
        rewardSet=trainCfg["rewardSet"],
        terrainSet=trainCfg["terrainSet"],
        historyLength=trainCfg["historyLength"],
        domainRand=args.domainRand,
        numEnvs=args.numEnvs,
        terrainOverride=evalTerrain(terrainName, difficulty),
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
    # Pollen's commands can randomly zero some envs (stand still). every eval duck should walk
    if hasattr(twist, "zero_command_prob"):
        twist.zero_command_prob = 0.0
    twist.resampling_time_range = (args.seconds + 1.0, args.seconds + 1.0)
    return ManagerBasedRlEnv(cfg, device=args.device)


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
    parser.add_argument("--terrains", nargs="+", default=list(EVAL_TERRAINS))
    parser.add_argument("--difficulties", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--minSpeed", type=float, default=0.1)
    parser.add_argument("--maxSpeed", type=float, default=0.4)
    parser.add_argument("--numEnvs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--pushes", action="store_true", help="keep random pushes on during eval")
    parser.add_argument("--domainRand", action="store_true", help="randomize mass/friction etc. during eval")
    parser.add_argument("--out", default="results/eval.csv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for ckptPath in args.checkpoints:
        actor, trainCfg = loadActor(ckptPath, args.device)
        for terrainName in args.terrains:
            for difficulty in args.difficulties:
                env = buildEvalEnv(trainCfg, terrainName, difficulty, args)
                result = runEpisode(env, actor)
                env.close()

                for i in range(len(result["fell"])):
                    rows.append(
                        {
                            "checkpoint": ckptPath,
                            "runName": trainCfg["runName"],
                            "terrain": terrainName,
                            "heldOut": terrainName in HELD_OUT,
                            "tier": TIERS[terrainName],
                            "difficulty": difficulty,
                            "commandSpeed": round(result["command"][i].item(), 4),
                            "fell": int(result["fell"][i]),
                            "velError": round(result["velError"][i].item(), 4),
                            "distance": round(result["distance"][i].item(), 4),
                        }
                    )
                fallRate = result["fell"].float().mean().item()
                velErr = result["velError"].mean().item()
                print(
                    f"{trainCfg['runName']:>16}  {terrainName:>16}  {TIERS[terrainName]:>6}  d={difficulty:.2f}  "
                    f"fall {fallRate:6.1%}  velErr {velErr:.3f} m/s"
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()