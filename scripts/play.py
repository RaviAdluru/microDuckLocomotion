"""Watch a trained checkpoint walk in the MuJoCo viewer.
    uv run scripts/play.py runs/<run>/model_3000.pt                  # native window
    uv run scripts/play.py runs/<run>/model_3000.pt --viewer viser   # in the browser (good over ssh)
"""

import argparse

import torch
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from duckWalk.envConfig import makeEnv
from duckWalk.terrains import evalTerrain
from duckWalk.trainer import loadActor


class Policy:
    # the viewer calls this with the full observation dict every control step
    def __init__(self, actor):
        self.actor = actor

    @torch.no_grad()
    def __call__(self, obs):
        return self.actor(obs["actor"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--viewer", choices=["native", "viser"], default="native")
    parser.add_argument("--numEnvs", type=int, default=1)
    parser.add_argument("--terrain", default="", help="show one eval terrain, e.g. stairs or waves")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    actor, trainCfg = loadActor(args.checkpoint, args.device)
    terrainOverride = evalTerrain(args.terrain, args.difficulty) if args.terrain else None

    env = makeEnv(
        device=args.device,
        rewardSet=trainCfg["rewardSet"],
        terrainSet=trainCfg["terrainSet"],
        historyLength=trainCfg["historyLength"],
        domainRand=False,
        numEnvs=args.numEnvs,
        play=True,
        terrainOverride=terrainOverride,
    )
    env = RslRlVecEnvWrapper(env)

    viewer = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
    viewer(env, Policy(actor)).run()


if __name__ == "__main__":
    main()
