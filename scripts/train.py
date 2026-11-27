"""Train a walking policy from scratch with our PPO.
Every field in duckWalk/config.py is a flag, e.g.
    uv run scripts/train.py --runName minimalFlat --rewardSet minimal --terrainSet flat
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from duckWalk.cliArgs import addConfigArgs, configFromArgs
from duckWalk.envConfig import makeEnv
from duckWalk.runLogger import RunLogger
from duckWalk.trainer import train


def main():
    parser = addConfigArgs(argparse.ArgumentParser())
    cfg = configFromArgs(parser.parse_args())

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("no CUDA device found, falling back to cpu (this will be very slow)")
        cfg.device = "cpu"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runDir = os.path.join("runs", f"{stamp}_{cfg.runName}")

    env = makeEnv(
        device=cfg.device,
        rewardSet=cfg.rewardSet,
        terrainSet=cfg.terrainSet,
        historyLength=cfg.historyLength,
        domainRand=cfg.domainRand,
        numEnvs=cfg.numEnvs,
        actionRateRamp=cfg.actionRateRamp,
        trackLinWeight=cfg.trackLinWeight,
        stepsPerIteration=cfg.numStepsPerEnv,
    )
    env.seed(cfg.seed)

    logger = RunLogger(cfg.logger, runDir, cfg)
    try:
        train(env, cfg, logger, runDir)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
