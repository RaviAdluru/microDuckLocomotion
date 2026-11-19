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
  sanityCheck.py         runs our PPO on cartpole to make sure the algorithm works
  train.py               train a walking policy
  play.py                watch a checkpoint in the MuJoCo viewer
  evaluate.py            test checkpoints on every eval terrain, writes a CSV
  plotEval.py            bar charts from the eval CSV
  trainLonestar6.slurm   batch job for a full run on TACC
```

## Setup

Tested on Linux (CachyOS) and on TACC Lonestar6. macOS and Windows are untested. On Windows, use WSL.

### 1. What you need

- **git**
- **[uv](https://docs.astral.sh/uv/)**, which manages Python and every package. You don't need Python installed yourself; uv downloads 3.12.
- **Disk space:** about 2 GB with CPU-only PyTorch, about 6 GB with the CUDA build. Check with `df -h ~`.
- **An NVIDIA GPU, only for real training.** Everything else (development, quick tests, the viewer, evaluation) runs fine on a laptop CPU. Full training runs go on TACC.

Install uv with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or through your package manager, e.g. `sudo pacman -Syu uv` on Arch/CachyOS. Open a new terminal afterwards and check that `uv --version` works.

### 2. Clone and install

```bash
git clone https://github.com/RaviAdluru/microDuckLocomotion.git
cd microDuckLocomotion
```

Pick one, depending on your machine:

```bash
uv sync --extra cpu    # no NVIDIA GPU (most laptops)
uv sync --extra cuda   # NVIDIA GPU (TACC, or a desktop with a CUDA card)
```

The first sync takes a few minutes. It creates `.venv/` in the repo and installs the exact versions in `uv.lock`, including microduck_rl and the BAM actuator model straight from GitHub. Always pass one of the two extras. A plain `uv sync` pulls the CUDA build by default.

### 3. Check it works

```bash
# robot config loads (should print "duck config ok" with no "Failed to load task package" warning)
uv run python -c "from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg; make_microduck_velocity_env_cfg(rough=True); print('duck config ok')"

# PPO learns on cartpole: the return column should climb over a few hundred iterations
uv run scripts/sanityCheck.py

# a tiny duck training run: should print "actor obs 51, critic obs 66, actions 14" then an iter 0 line
uv run scripts/train.py --device cpu --numEnvs 8 --maxIterations 2 --logger none --runName smokeTest
```

The first run of anything compiles a bunch of Warp kernels (lots of `Module ... took ... ms (compiled)` lines). They get cached in `~/.cache/warp`, so later runs start much faster.

### 4. Logging

For wandb, make a free account at wandb.ai, then run `uv run wandb login` once. If you'd rather skip wandb, add `--logger tensorboard` to any training command.

## Why the versions are pinned

Don't bump these in `pyproject.toml` without testing first. Each pin fixes a real breakage:

- **`better-actuator-models` (BAM) at commit `62bd8ce`.** The BAM branch microduck_rl points at has moved on, and the newer version breaks microduck's actuator config (`unexpected keyword argument 'vin_drop_gain_range'`). It's pinned through `override-dependencies`, because a normal source entry conflicts with microduck_rl's own pin.
- **`mjviser==0.0.14` / `viser==1.0.30`.** Newer mjviser pulls in `mujoco-core` 4.x, which overwrites files from the `mujoco` 3.10 package the simulator needs (`libmujoco.so.3.10.0: cannot open shared object file`).
- **`mujoco`, `mujoco-warp`, `mjlab`, `warp-lang`, `torch`.** These match what microduck_rl was tested with.
- **The CPU torch index is limited to x86_64**, so it doesn't clash with microduck_rl's own torch rule for ARM machines.

## TACC (Lonestar6)

Use Lonestar6's A100 nodes. Don't use Vista: its GPU nodes are ARM, and our CUDA setup only covers x86.

One-time setup on a login node:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export UV_CACHE_DIR=$SCRATCH/uvCache' >> ~/.bashrc   # $HOME quota is small, keep uv's cache in scratch
source ~/.bashrc
cd $WORK
git clone https://github.com/RaviAdluru/microDuckLocomotion.git
cd microDuckLocomotion
uv sync --extra cuda
```

To test interactively on a GPU node (max 2 hours on the dev queue):

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00
uv run scripts/train.py --runName gpuTest --maxIterations 50 --logger tensorboard
```

For full runs, open `scripts/trainLonestar6.slurm`, put our allocation name on the `#SBATCH -A` line, then submit from the repo root. Anything after the script name goes straight to `train.py`:

```bash
sbatch scripts/trainLonestar6.slurm --runName baseline
sbatch scripts/trainLonestar6.slurm --runName history5 --historyLength 5
squeue -u $USER              # check on your jobs
tail -f slurm_<jobid>.out    # follow the output
```

Compute nodes may not reach wandb, so the slurm script logs to tensorboard. To look at results or watch the robot, copy the run folder back to your laptop:

```bash
scp -r <tacc-user>@ls6.tacc.utexas.edu:/path/to/microDuckLocomotion/runs/<run> runs/
```

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
| `--seed` | `1` | change it to repeat a run with different randomness |
| `--logger` | `wandb` | `wandb`, `tensorboard` or `none` |
| `--device` | `cuda:0` | falls back to cpu automatically if there's no GPU |

For scale: a full run is 3000 × 24 × 4096 ≈ 295 million sim steps. That's a couple of hours on an A100 and weeks on a laptop CPU (a laptop does roughly 50-80 steps per second with 8 robots).

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

On a full 4096-env run, `train/approxKl` should sit around 0.01. Tiny test runs (8 envs) show much bigger values, which is just noise from the small batch.

## Watching the robot

```bash
uv run scripts/play.py runs/<run>/model_3000.pt
uv run scripts/play.py runs/<run>/model_3000.pt --terrain stairs --difficulty 1.0
uv run scripts/play.py runs/<run>/model_3000.pt --viewer viser
```

`--viewer viser` serves the sim in the browser instead of opening a window. Playback runs fine on a laptop CPU.

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
- The warning `Actuator config matched 14 joint(s) but the same expressions also match 7 site(s)` comes from Pollen's robot config and is harmless.
- The Microduck model files are CC BY-NC-SA 4.0 (non-commercial) per microduck_rl. That's fine for coursework. Credit Pollen Robotics if you reuse anything.

## Troubleshooting

| error | fix |
|---|---|
| `Unknown command: uv` / `uv: command not found` | uv isn't installed or isn't on your PATH. Install it, then open a new terminal. |
| `No space left on device` during sync | Free space (`uv cache clean`, clear your package manager cache), or use `--extra cpu`, which is about 4 GB smaller. |
| `libmujoco.so.3.10.0: cannot open shared object file` | Something pulled in `mujoco-core` 4.x. Check the mjviser pin, then `rm -rf .venv && uv sync --extra cpu`. |
| `Failed to load task package mjlab_microduck: ... vin_drop_gain_range` | Wrong BAM version. Check the BAM override pin, then re-sync. |
| `conflicting indexes for package torch` | Don't add torch back to the main `dependencies` list, and keep the `marker` on the CPU torch source. |
| `Warp CUDA warning: Could not find or load the NVIDIA CUDA driver` | Expected on a machine without an NVIDIA GPU. Everything falls back to CPU. |

## Working on this together

Ravi adds collaborators from the repo's Settings → Collaborators page (or `gh api -X PUT repos/RaviAdluru/microDuckLocomotion/collaborators/<username>`). Accept the email invite, then clone as above.

Work on a branch and open a PR instead of pushing to `main`:

```bash
git checkout -b yourName/whatYouAreDoing
# ...changes...
git push -u origin yourName/whatYouAreDoing
gh pr create          # or open the PR on github.com
```

Don't commit `runs/`, `wandb/`, slurm logs or eval output, they're in `.gitignore`. Share checkpoints through wandb or a shared drive instead. If you change dependencies, commit the updated `uv.lock` in the same PR.
