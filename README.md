# microDuckLocomotion

Teaching the [Microduck](https://pollen-robotics.com/microduck) (a 25 cm bipedal robot from Pollen Robotics / Hugging Face) to walk from scratch with reinforcement learning, using our own PPO implementation instead of an off-the-shelf trainer.

The policy is trained in MuJoCo across a mix of terrains (flat ground, bumpy grids, smooth rolling bumps, slopes). It never gets a map of the terrain. It only sees what the real robot could measure: IMU, joint encoders, its last action and the velocity command. Then we test it on terrains it never saw during training to check how well it generalizes.

Class project for Intro to Machine Learning at UT Austin, Fall 2026.

## What's ours and what's borrowed

We use Pollen's [microduck_rl](https://github.com/pollen-robotics/microduck_rl) for the robot itself: the MuJoCo model, the servo (BAM) actuator model, sensors and sim settings. Physics runs on [mjlab](https://github.com/mujocolab/mjlab) with MuJoCo Warp.

Everything on the learning side is written here:

- PPO (clipped surrogate loss, clipped value loss, entropy bonus, GAE, adaptive learning rate)
- actor and critic networks with running observation normalization
- the reward function, split into sets so we can ablate it
- training and held-out terrain sets with a difficulty curriculum
- evaluation and plotting

Pollen's own walking task (`Mjlab-Velocity-Rough-MicroDuck`) is a good upper-bound baseline to compare against, since it has months of reward tuning behind it.

## Layout

```
duckWalk/
  config.py         every training setting, all exposed as command line flags
  envConfig.py      builds the env: Pollen's robot + our rewards/obs/terrain/randomization
  rewards.py        reward terms, grouped into minimal / posture / full
  terrains.py       training terrain mix and the held-out eval terrains
  networks.py       Gaussian actor, critic, running normalizer
  rolloutBuffer.py  rollout storage and GAE
  ppo.py            the PPO loss and update
  trainer.py        collect -> update -> log -> checkpoint loop
  runLogger.py      wandb / tensorboard / no logging
  cliArgs.py        turns config.py into argparse flags
scripts/
  sanityCheck.py    runs our PPO on cartpole to make sure the algorithm works
  train.py          train a walking policy
  play.py           watch a checkpoint in the MuJoCo viewer
  evaluate.py       test checkpoints on every eval terrain, writes a CSV
  plotEval.py       bar charts from the eval CSV
```

## Setup

You need an NVIDIA GPU for real training (the sim runs thousands of robots in parallel on the GPU). CPU works for checking that things run, but it's orders of magnitude slower (about 80 sim steps per second on a laptop CPU with 8 robots).

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the repo root:

```bash
uv sync
```

This pulls Python 3.12, mjlab, PyTorch, and microduck_rl plus the BAM actuator model straight from GitHub at pinned commits. The first sync takes a few minutes. It also writes `uv.lock`, which should be committed so we both get identical versions.

If you're using wandb for logging, run `uv run wandb login` once.

### TACC

On a Lonestar6 GPU node the same `uv sync` works. Vista's GPU nodes are ARM (aarch64), where the PyPI PyTorch wheel is CPU-only, so stick to Lonestar6 unless we add the CUDA wheel index for ARM. Keep the repo and `.venv` in `$WORK` or `$SCRATCH`, not `$HOME`, since the environment is a few GB. Use `--logger tensorboard` if the compute node can't reach wandb.

## Quick check

Before training the duck, make sure PPO itself learns something:

```bash
uv run scripts/sanityCheck.py
```

On cartpole the return should climb steadily over the first couple hundred iterations (on CPU with 128 envs it went from about 20 to about 49 in 200 iterations). If it doesn't, the problem is in `ppo.py`, not the robot.

## Training

```bash
uv run scripts/train.py --runName baseline
```

Defaults: 4096 envs, full reward set, mixed terrain, domain randomization on, no observation history, 3000 iterations. Every field in `duckWalk/config.py` can be changed from the command line:

```bash
uv run scripts/train.py --runName minimalFlat --rewardSet minimal --terrainSet flat
uv run scripts/train.py --runName history5 --historyLength 5
uv run scripts/train.py --runName noDr --domainRand false
uv run scripts/train.py --runName quickTest --numEnvs 64 --maxIterations 20 --logger none --device cpu
```

Checkpoints go to `runs/<timestamp>_<runName>/model_<iter>.pt` every 250 iterations. Pick up a stopped run with `--resume runs/.../model_1000.pt`.

Useful flags:

| flag | default | what it does |
|---|---|---|
| `--rewardSet` | `full` | `minimal`, `posture` or `full`, see `rewards.py` |
| `--terrainSet` | `mixed` | `flat` or `mixed` |
| `--historyLength` | `1` | how many past frames of actor obs the policy sees |
| `--domainRand` | `true` | randomize mass, friction, motor gains, CoM, plus random pushes |
| `--numEnvs` | `4096` | parallel robots |
| `--maxIterations` | `3000` | PPO iterations (each one is 24 steps from every env) |
| `--logger` | `wandb` | `wandb`, `tensorboard` or `none` |
| `--device` | `cuda:0` | falls back to cpu automatically if there's no GPU |

## Watching training

With wandb, curves show up at wandb.ai under the project `microDuckLocomotion`. With tensorboard:

```bash
uv run tensorboard --logdir runs
```

What gets logged:

- `episode/return`, `episode/length`: averaged over the last 200 finished episodes
- `env/Episode_Reward/<term>`: each reward term on its own, the most useful plot for debugging reward design
- `env/Curriculum/terrain_levels/<terrain>`: how far up the difficulty ladder the ducks are on each terrain type
- `env/Episode_Termination/*`: why episodes end (fell over vs timed out)
- `train/*`: policy loss, value loss, entropy, approx KL, clip fraction, explained variance, learning rate, action std

## Watching the robot

```bash
uv run scripts/play.py runs/<run>/model_3000.pt
uv run scripts/play.py runs/<run>/model_3000.pt --terrain stairs --difficulty 1.0
uv run scripts/play.py runs/<run>/model_3000.pt --viewer viser
```

`--viewer viser` serves the sim in the browser, which is the way to go over ssh or on TACC.

## Evaluation

`evaluate.py` runs each checkpoint on every eval terrain at the given difficulties. Every env gets a fixed straight-ahead speed command between `--minSpeed` and `--maxSpeed`. Pushes and randomization are off unless you pass `--pushes` / `--domainRand`.

```bash
uv run scripts/evaluate.py runs/a/model_3000.pt runs/b/model_3000.pt --out results/eval.csv
uv run scripts/plotEval.py results/eval.csv --difficulty 1.0
```

The CSV has one row per robot: terrain, whether it was held out, difficulty, commanded speed, whether it fell, mean velocity error and distance walked.

| terrain | in training? | notes |
|---|---|---|
| flat | yes | |
| bumpyGrid | yes | random-height boxes, up to 1 cm |
| smoothBumps | yes | perlin noise, up to 1.5 cm |
| slope | yes | up to about 5.7 deg |
| waves | no | smooth sine waves, new shape |
| stairs | no | pyramid stairs, up to 1.5 cm steps |
| bumpyGridHard | no | same type, up to 2 cm |
| smoothBumpsHard | no | same type, up to 3 cm |
| slopeHard | no | up to about 10 deg |

The heights look tiny but the duck only lifts its feet 1-2 cm, so these are real obstacles at its scale.

## Planned experiments

1. **Reward ablation.** Train `minimal`, `posture` and `full` on flat ground and compare gaits. `minimal` will probably find some weird shuffle or hop, which is the point.
2. **Terrain.** Train on `flat` vs `mixed`, then evaluate both on everything. This shows what the terrain curriculum buys on held-out terrain.
3. **Memory.** `--historyLength 1` vs `5` vs `10`. The policy is blind, so a short history should help it feel out the ground.
4. **Domain randomization.** On vs off, evaluated with `--pushes --domainRand` to see robustness.
5. **Baseline.** Compare against Pollen's trained rough-terrain policy.

Run each with a few seeds (`--seed`) so the comparisons hold up.

## Notes

- Actor obs is 51 numbers per frame (base angular velocity, gravity direction, 14 joint positions, 14 joint velocities, last action, velocity command). The critic gets 66, adding true base velocity and foot contact info that only exists in sim.
- The action is 14 joint position targets at 50 Hz. Physics runs at 200 Hz.
- Pollen's config also commands head and body poses. We pin those to zero and keep a `headHold` reward so the neck motors stay still.
- The Microduck model files are CC BY-NC-SA 4.0 (non-commercial) per microduck_rl. That's fine for coursework. Credit Pollen Robotics if you reuse anything.

## Working on this together

Work on a branch and open a PR instead of pushing to `main`:

```bash
git checkout -b yourName/whatYouAreDoing
# ...changes...
git push -u origin yourName/whatYouAreDoing
```

Don't commit `runs/`, `wandb/` or eval output, they're in `.gitignore`. Share checkpoints through wandb or a shared drive instead.
