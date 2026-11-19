"""Runs our PPO on mjlab's cartpole balance task to make sure the algorithm itself works.
If return doesn't climb here within a few hundred iterations, something is wrong in ppo.py, not the duck.
    uv run scripts/sanityCheck.py --device cpu
"""

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from duckWalk.config import TrainConfig
from duckWalk.runLogger import RunLogger
from duckWalk.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--numEnvs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--logger", default="none")
    args = parser.parse_args()

    torch.manual_seed(0)
    envCfg = load_env_cfg("Mjlab-Cartpole-Balance")
    envCfg.scene.num_envs = args.numEnvs
    env = ManagerBasedRlEnv(envCfg, device=args.device)

    cfg = TrainConfig(
        runName="cartpoleSanity",
        device=args.device,
        numEnvs=args.numEnvs,
        maxIterations=args.iterations,
        hiddenDims=(64, 64),
        logger=args.logger,
        saveInterval=10**9,
    )
    logger = RunLogger(cfg.logger, "runs/cartpoleSanity", cfg)
    try:
        train(env, cfg, logger, "runs/cartpoleSanity")
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
