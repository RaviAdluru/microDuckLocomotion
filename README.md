# microDuckLocomotion

Teaching the [Microduck](https://pollen-robotics.com/microduck) (a 25 cm bipedal robot from Pollen Robotics / Hugging Face) to walk from scratch with reinforcement learning, using our own PPO implementation instead of an off-the-shelf trainer.

The policy is trained in MuJoCo across a mix of terrains (flat ground, bumpy grids, smooth rolling bumps, slopes). It never gets a map of the terrain. It only sees what the real robot could measure: IMU, joint encoders, its last action and the velocity command. Then we test it on terrains it never saw during training to check how well it generalizes.

Class project for Intro to Machine Learning at UT Austin, Fall 2026.

## Contents

1. [What's ours and what's borrowed](#whats-ours-and-whats-borrowed)
2. [Layout](#layout)
3. [Setup](#setup)
4. [The basic loop](#the-basic-loop)
5. [Training](#training)
6. [Changing things](#changing-things): rewards, terrains, observations, commands, networks, PPO settings, logging
7. [Watching training](#watching-training): what each graph means
8. [Making your own graphs](#making-your-own-graphs)
9. [Watching the robot](#watching-the-robot)
10. [Evaluation](#evaluation)
11. [Planned experiments](#planned-experiments)
12. [Why the versions are pinned](#why-the-versions-are-pinned)
13. [Troubleshooting](#troubleshooting)
14. [Working on this together](#working-on-this-together)

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
  evaluate.py       test checkpoints on the eval terrains (validation or test split), writes a CSV
  plotEval.py       bar charts per terrain from the eval CSV, with error bars
  plotValidation.py validation curve: eval results vs training iteration
runs/               training output (checkpoints, tensorboard logs), not committed
results/            eval CSVs and plots, not committed
```

## Setup

Tested on Linux (CachyOS). macOS and Windows are untested. On Windows, use WSL.

### 1. What you need

- **git**
- **[uv](https://docs.astral.sh/uv/)**, which manages Python and every package. You don't need Python installed yourself; uv downloads 3.12.
- **Disk space:** about 2 GB with CPU-only PyTorch, about 6 GB with the CUDA build. Check with `df -h ~`.
- **An NVIDIA GPU, only for full training runs.** Everything else (development, quick tests, the viewer, evaluation, graphs) runs fine on a laptop CPU.

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
uv sync --extra cuda   # NVIDIA GPU
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

For wandb, make a free account at wandb.ai, then run `uv run wandb login` once. If you'd rather skip wandb, add `--logger tensorboard` to any training command. Everything in this README works with either one.

## The basic loop

Almost everything in this project is the same four steps:

1. **Change something:** a reward, a terrain, a setting (see [Changing things](#changing-things)).
2. **Train:** `uv run scripts/train.py --runName whatYouChanged ...`
3. **Look at the graphs** while it trains (see [Watching training](#watching-training)).
4. **Watch the robot** walk with a saved checkpoint (see [Watching the robot](#watching-the-robot)), then **evaluate** it on held-out terrain (see [Evaluation](#evaluation)).

Give every run a `--runName` that says what's different about it. The output folder, the wandb run and the eval CSV all use that name, so `minimalFlat` beats `test3`.

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

Checkpoints go to `runs/<timestamp>_<runName>/model_<iter>.pt` every 250 iterations (change with `--saveInterval`). Pick up a stopped run with `--resume runs/.../model_1000.pt`.

All the flags:

| flag | default | what it does |
|---|---|---|
| `--runName` | `baseline` | name for the output folder and the logs |
| `--seed` | `1` | change it to repeat a run with different randomness |
| `--device` | `cuda:0` | falls back to cpu automatically if there's no GPU |
| `--numEnvs` | `4096` | parallel robots |
| `--rewardSet` | `full` | `minimal`, `posture` or `full`, see `rewards.py` |
| `--terrainSet` | `mixed` | `flat` or `mixed`, see `terrains.py` |
| `--historyLength` | `1` | how many past frames of actor obs the policy sees |
| `--domainRand` | `true` | randomize mass, friction, motor gains, CoM, plus random pushes |
| `--hiddenDims` | `512 256 128` | hidden layer sizes for actor and critic |
| `--initStd` | `1.0` | starting exploration noise on actions |
| `--maxIterations` | `3000` | PPO iterations (each one is `numStepsPerEnv` steps from every env) |
| `--numStepsPerEnv` | `24` | rollout length per iteration |
| `--learningRate` | `1e-3` | starting learning rate (adapted during training, see below) |
| `--gamma` | `0.99` | discount factor |
| `--lam` | `0.95` | GAE lambda |
| `--clipRatio` | `0.2` | PPO clip range |
| `--valueCoef` | `1.0` | weight on the value loss |
| `--entropyCoef` | `0.01` | weight on the entropy bonus |
| `--numEpochs` | `5` | passes over each rollout per update |
| `--numMiniBatches` | `4` | minibatches per pass |
| `--targetKl` | `0.01` | KL the adaptive learning rate aims for |
| `--maxGradNorm` | `1.0` | gradient clipping |
| `--logger` | `wandb` | `wandb`, `tensorboard` or `none` |
| `--wandbProject` | `microDuckLocomotion` | wandb project name |
| `--saveInterval` | `250` | save a checkpoint every N iterations |
| `--resume` | | checkpoint path to keep training from |

For scale: a full run is 3000 × 24 × 4096 ≈ 295 million sim steps. That's a couple of hours on a modern NVIDIA GPU and weeks on a laptop CPU (a laptop does roughly 50-80 steps per second with 8 robots). On a laptop, stick to small `--numEnvs` and `--maxIterations` for checking that a change runs.

## Changing things

This section goes file by file: what each part does, the usual things you'd change, and how. For a quick experiment, a command line flag is usually enough. Edit the code when you want to change what a setting means or add something new.

### Training defaults: `duckWalk/config.py`

`TrainConfig` holds every setting, and `cliArgs.py` turns each field into a `--flag` automatically. So:

- **To change a default for good**, edit the value in `config.py`.
- **To change it for one run**, pass the flag. You don't need to touch the file.
- **To add a new setting**, add a field to `TrainConfig` (e.g. `maxSpeed: float = 0.4`). It becomes `--maxSpeed` right away. Then read it where you need it, usually `cfg.maxSpeed` in `scripts/train.py` or `trainer.py`. Use it in `envConfig.py` by adding it as a `makeEnvCfg` argument and passing it from `train.py`.

### Rewards: `duckWalk/rewards.py`

This is where most of the interesting changes happen. The reward is a weighted sum of terms. Each term is a `RewardTermCfg`:

```python
"upright": RewardTermCfg(
    func=velMdp.upright,     # function that computes the term for every env
    weight=2.0,              # positive = reward, negative = penalty
    params={"std": ...},     # extra arguments passed to func
),
```

The sets build on each other: `minimalRewards()` has velocity tracking plus staying alive, `postureRewards()` adds upright/pose/smoothness terms, and `fullRewards()` adds gait terms (air time, foot clearance, slip, self collision).

**Change a weight:** edit the `weight=` number. Rule of thumb: if the duck ignores something, raise its weight. If it's too stiff or scared to move, lower the penalties.

**Remove a term:** delete its entry, or comment it out.

**Add a new reward set:** write a function that returns a dict of terms, then register it at the bottom of the file:

```python
def myRewards():
    rewards = postureRewards()
    rewards["airTime"] = fullRewards()["airTime"]   # posture + only the air time term
    return rewards

REWARD_SETS = {
    "minimal": minimalRewards,
    "posture": postureRewards,
    "full": fullRewards,
    "mine": myRewards,          # now usable as --rewardSet mine
}
```

**Write your own reward term:** any function that takes `env` and returns one number per env (a tensor of shape `[numEnvs]`). For example, a penalty on sideways sliding:

```python
def lateralVelocity(env):
    robot = env.scene["robot"]
    return robot.data.root_link_lin_vel_b[:, 1] ** 2   # body-frame y velocity squared

# then add it to a set:
rewards["lateralVel"] = RewardTermCfg(func=lateralVelocity, weight=-1.0)
```

Useful things to build terms from: `robot.data.root_link_lin_vel_b` (base velocity, body frame), `robot.data.root_link_ang_vel_b`, `robot.data.joint_pos`, `robot.data.joint_vel`, and `env.command_manager.get_command("twist")` (the commanded velocity). The ready-made functions in `mjlab.envs.mdp`, `mjlab.tasks.velocity.mdp` and `mjlab_microduck.tasks.mdp` are worth reading before writing your own.

Every term gets its own graph automatically (`env/Episode_Reward/<termName>`), so name terms clearly.

### Terrains: `duckWalk/terrains.py`

Four parts:

- **Terrain types:** `flat()`, `bumpyGrid()`, `smoothBumps()`, `slope()`, `waves()`, `stairsDown()`, `stairsUp()`. Each takes a height or slope limit plus a few shape settings. Heights are in meters and should stay small: the duck lifts its feet 1-2 cm.
- **The training mix:** `trainTerrains("mixed")`. Change `subTerrains` to pick which types are used and `proportions` to change how many ducks train on each (they should add up to 1). Each duck keeps one terrain type for the whole run; only its difficulty row changes. `num_rows` is the number of difficulty levels the curriculum moves through. The heights you pass are the max, reached at the hardest row.
- **Eval terrains:** `EVAL_TERRAINS` (single-type 8 m tiles) and `COMBINED_COURSES` (2 m patchwork tiles). `TIERS` labels each one `seen`, `harder`, `novel` or `combined`, and `SPLITS` groups them into validation and test (see [Evaluation](#evaluation)).
- **`TRAINING_TYPE`** maps each seen/harder eval terrain to the training type it's a version of, so evaluation can check whether training actually reached that difficulty.

**Make training terrain harder:** raise the limits inside `trainTerrains`, e.g. `bumpyGrid(maxHeight=0.015)`.

**Add a new training mix:** add another `if name == "...":` branch to `trainTerrains`, then train with `--terrainSet yourName`.

**Add an eval terrain:** add it to `EVAL_TERRAINS` and give it a tier in `TIERS` (there's an `assert` that fails if you forget). If it's a version of a training type, add it to `TRAINING_TYPE` too. If you add a terrain type to training, move its eval version out of `novel`, or the held-out results stop meaning anything.

**Watch the collision count.** Box terrains create one collision box per step or cell. Shrinking bumpyGrid cells from 0.45 m to 0.175 m raised the terrain from about 2.6k to 17.7k geoms and ran out of memory with just 16 ducks on CPU. Heightfield terrains (`Hf...`) are a single collision object per tile, so use those for fine-grained roughness. The startup line `softened N terrain geoms` shows the count.

Other terrain types mjlab has (pits, rails, stepping stones, gaps and so on) are in `mjlab.terrains`. Look at the `...TerrainCfg` classes there.

### Observations, commands, randomization: `duckWalk/envConfig.py`

`makeEnvCfg` starts from Pollen's walking config and swaps in our pieces. Things you'd change here:

**Observation history.** Set with `--historyLength N`. The actor then sees its last N frames instead of only the current one.

- **Why:** the actor is blind. One frame can't separate "flat ground" from "the left foot just landed on a 1 cm ledge," but a few frames of how the joints and body responded can. History lets the policy infer things it can't measure directly: terrain under the feet, contact state, its own velocity.
- **How it's built:** mjlab keeps a ring buffer for every actor term and flattens it oldest-to-newest, term by term. With N = 5, `joint_pos` becomes 70 numbers (5 × 14) and the full actor input is 255 (5 × 51). The startup table shows it as `(70,) ← 5×(14,)`.
- **Resets:** when an episode ends, that duck's buffer is filled with the first frame of its new episode, so nothing leaks between episodes.
- **Only the actor gets it.** The critic already sees the privileged state the actor is trying to infer.
- **Cost:** the first layer grows with N. With the default `512 256 128` network, the actor goes from about 193k parameters at N = 1 to about 297k at N = 5. More inputs also means more ways to overfit to the training terrain, which is why N should be chosen on validation results.
- **Everything downstream follows automatically.** The checkpoint stores the history length, and `play.py` and `evaluate.py` rebuild the env to match.

**What the networks observe.** The actor and critic terms are in `cfg.observations["actor"].terms` and `cfg.observations["critic"].terms`. Remove one with `.pop("termName")`. For example, to test whether the policy needs the last action:

```python
cfg.observations["actor"].terms.pop("actions")
```

The term names are printed in the table at the start of every run.

**Speed commands.** The duck is trained to follow random velocity commands. To change the range, add lines like these to `makeEnvCfg`:

```python
cfg.commands["twist"].ranges.lin_vel_x = (-0.1, 0.4)   # forward/back, m/s
cfg.commands["twist"].ranges.lin_vel_y = (-0.1, 0.1)   # sideways, m/s
cfg.commands["twist"].ranges.ang_vel_z = (-1.0, 1.0)   # turning, rad/s
```

**Domain randomization.** `DOMAIN_RAND_EVENTS` lists what gets turned off with `--domainRand false`. To test one source of randomization at a time, pop single events instead, e.g. `cfg.events.pop("push_robot", None)`.

**Episode length.** `cfg.episode_length_s = 20.0` (seconds).

### Networks: `duckWalk/networks.py`

The actor is an MLP that outputs the mean of a Gaussian over the 14 joint targets, plus a learned standard deviation (the exploration noise). The critic is an MLP that outputs one value. Both normalize their inputs with running statistics that get saved in the checkpoint.

- **Layer sizes:** `--hiddenDims 256 128` (no code change).
- **Activation:** change `nn.ELU()` in `mlp()` to `nn.Tanh()` or `nn.ReLU()`.
- **Starting noise:** `--initStd 0.5` explores less at the start.

### PPO: `duckWalk/ppo.py`, `duckWalk/rolloutBuffer.py`, `duckWalk/trainer.py`

`ppo.py` holds the loss: the clipped surrogate (policy) loss, the clipped value loss and the entropy bonus. Each term is commented in place. `rolloutBuffer.py` computes advantages with GAE. `trainer.py` runs the loop and decides what gets logged.

All the hyperparameters are flags. Here's what to try when something looks off:

| symptom | try |
|---|---|
| return climbs then collapses | lower `--learningRate` or `--targetKl`, or raise `--numStepsPerEnv` |
| learning is very slow and `train/approxKl` stays tiny | raise `--learningRate` or `--targetKl` |
| `train/actionStd` drops to near zero early and progress stops | raise `--entropyCoef` |
| `train/actionStd` stays high and the duck stays jittery | lower `--entropyCoef` |
| `train/explainedVariance` stays near 0 | the critic isn't learning: raise `--valueCoef` or the critic size |

The learning rate adapts itself: after each minibatch it's divided by 1.5 if KL is above 2 × `targetKl`, and multiplied by 1.5 if KL is below half of it (`adaptLearningRate` in `ppo.py`).

### Logging a new metric: `duckWalk/trainer.py`

Anything you add to the `logValues` dict in `train()` shows up in wandb or tensorboard with no other changes. For example, to log mean base height:

```python
logValues["robot/baseHeight"] = env.scene["robot"].data.root_link_pos_w[:, 2].mean().item()
```

Put it after the other `logValues[...]` lines and before `logger.log(logValues, iteration)`.

## Watching training

With wandb, open wandb.ai and find the project `microDuckLocomotion`. Every run shows up there with its config, so you can filter and compare runs by setting.

With tensorboard, run this in a second terminal while training (or after):

```bash
uv run tensorboard --logdir runs
```

Then open http://localhost:6006. Checking several runs in the left panel overlays them on the same graphs.

What each graph tells you:

| graph | what it is | what you want to see |
|---|---|---|
| `episode/return` | total reward per episode, averaged over the last 200 episodes | climbing, then leveling off |
| `episode/length` | steps before an episode ends (1000 = full 20 s episode) | climbing toward the max as the duck stops falling |
| `env/Episode_Reward/<term>` | each reward term on its own | the main debugging plot. Shows which terms the duck is actually earning and which penalties dominate |
| `env/Episode_Termination/fell_over` | fraction of episodes ending in a fall | dropping toward 0 |
| `env/Episode_Termination/time_out` | fraction of episodes that ran the full length | rising toward 1 |
| `env/Curriculum/terrain_levels/<type>` | average difficulty level per terrain type | rising. Flat means the duck is stuck at that difficulty |
| `env/Metrics/twist/error_vel_xy` | velocity tracking error | dropping |
| `env/Metrics/air_time_mean`, `peak_height_mean`, `slip_velocity_mean` | gait stats: step timing, foot lift, foot sliding | useful for describing the gait in the report |
| `train/policyLoss`, `train/valueLoss` | the two loss terms | value loss should come down. Policy loss is noisy and doesn't trend much, which is normal for PPO |
| `train/approxKl` | how much the policy changed per update | around 0.01 on a full run. Tiny test runs show bigger values, which is just noise from the small batch |
| `train/clipFraction` | fraction of samples hitting the PPO clip | roughly 0.05-0.3 |
| `train/explainedVariance` | how well the critic predicts returns | well above 0, ideally 0.8+ |
| `train/actionStd` | exploration noise | slowly decreasing |
| `train/learningRate` | current learning rate after adaptation | moves around, shouldn't pin at the 1e-5 floor or 1e-2 ceiling |
| `perf/stepsPerSec` | sim speed | just for sanity |

## Making your own graphs

The wandb and tensorboard plots are fine for checking progress. For the report you'll want your own figures. Every run writes its logs into its `runs/` folder (when using `--logger tensorboard`), and you can read them straight into matplotlib. This plots a few metrics for every run side by side:

```python
# save as scripts/plotRuns.py, run with: uv run scripts/plotRuns.py
import glob

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tags = ["episode/return", "env/Episode_Reward/trackLinVel", "train/approxKl"]
runs = sorted(glob.glob("runs/*"))

fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4))
for runDir in runs:
    events = EventAccumulator(runDir)
    events.Reload()
    for ax, tag in zip(axes, tags):
        if tag in events.Tags()["scalars"]:
            points = events.Scalars(tag)
            ax.plot([p.step for p in points], [p.value for p in points], label=runDir.split("_", 2)[-1])
for ax, tag in zip(axes, tags):
    ax.set_title(tag)
    ax.set_xlabel("iteration")
axes[0].legend()
fig.tight_layout()
fig.savefig("results/trainingCurves.png", dpi=150)
```

Change `tags` to any graph name from the table above, and change the `glob` pattern to pick which runs to compare (e.g. `"runs/*history*"`). With wandb you can also download any graph as a PNG or CSV from its menu in the web UI.

For held-out evaluation results, `scripts/plotEval.py` makes bar charts from the eval CSV (see [Evaluation](#evaluation)). The CSV has one row per robot, so it's easy to load with pandas or the `csv` module for any other plot you want.

## Watching the robot

`play.py` loads a checkpoint and opens the MuJoCo viewer with the robot walking. It rebuilds the env with the same settings the checkpoint was trained with.

```bash
uv run scripts/play.py runs/<run>/model_3000.pt                              # native window
uv run scripts/play.py runs/<run>/model_3000.pt --terrain stairsUp --difficulty 1.0
uv run scripts/play.py runs/<run>/model_3000.pt --terrain combinedAll --difficulty 1.0 --numEnvs 8
uv run scripts/play.py runs/<run>/model_3000.pt --numEnvs 4                  # several ducks at once
uv run scripts/play.py runs/<run>/model_3000.pt --viewer viser               # in the browser instead
```

| flag | what it does |
|---|---|
| `--viewer` | `native` opens a MuJoCo window, `viser` serves the sim in the browser (it prints the address to open) |
| `--numEnvs` | how many ducks to show |
| `--terrain` | show one eval terrain instead of the training terrain, e.g. `waves`, `stairsUp`, `combinedAll` (any name in the [eval terrain table](#evaluation)) |
| `--difficulty` | 0 to 1, how hard that terrain is |
| `--device` | `cpu` or `cuda:0`. Playback runs fine on a laptop CPU |

Pushes and randomization are off during playback, so you see the policy itself.

**Seeing how the duck improves over training.** A checkpoint is saved every 250 iterations, so play them in order:

```bash
uv run scripts/play.py runs/<run>/model_0.pt      # untrained: flails and falls
uv run scripts/play.py runs/<run>/model_500.pt
uv run scripts/play.py runs/<run>/model_1500.pt
uv run scripts/play.py runs/<run>/model_3000.pt
```

Use a smaller `--saveInterval` (e.g. 100) when training if you want finer steps. The viewer shows saved checkpoints, not the run as it's training. To check on a run in progress, play its latest checkpoint while it keeps going.

Side-by-side playback across runs at the same iteration is the easiest way to see what a reward or terrain change did to the gait. Screen-record the viewer for the presentation.

## Evaluation

`evaluate.py` runs checkpoints on the eval terrains at fixed difficulties. Every duck gets a fixed straight-ahead speed command between `--minSpeed` and `--maxSpeed` (inside the training range of ±0.4 m/s) and walks with the deterministic (mean) action. Pushes and randomization are off unless you pass `--pushes` / `--domainRand`.

### Validation vs test

The eval terrains are split in two, and the split matters for keeping the results honest:

- **Validation** (`--split validation`): the `seen` and `harder` tiers. Look at these as much as you want while choosing checkpoints, reward weights, history length and so on.
- **Test** (`--split test`): the `novel` and `combined` tiers. Run these once, at the end, on the runs you already picked. Once you start tuning things to make the test numbers better, they stop measuring generalization.

### Commands

```bash
# validation curve over one training run: every saved checkpoint
uv run scripts/evaluate.py runs/<run>/model_*.pt --split validation --difficulties 0.5 --out results/val.csv
uv run scripts/plotValidation.py results/val.csv --difficulty 0.5

# final test, once, on the chosen checkpoints
uv run scripts/evaluate.py runs/a/model_3000.pt runs/b/model_3000.pt --split test --out results/test.csv
uv run scripts/evaluate.py runs/a/model_3000.pt runs/b/model_3000.pt --terrains combinedSeen combinedAll --seconds 20 --out results/testCombined.csv
uv run scripts/plotEval.py results/test.csv --difficulty 1.0

# robustness pass: same thing with pushes and randomized physics
uv run scripts/evaluate.py runs/a/model_3000.pt --split test --pushes --domainRand --out results/testRobust.csv
```

`plotValidation.py` plots fall rate and velocity error against training iteration, one line per terrain. Put it next to the training return curve: the gap between them is the generalization gap. `plotEval.py` makes one bar per run per terrain with standard-error bars, using each run's latest checkpoint in the CSV unless you pass `--iteration`.

| flag | default | what it does |
|---|---|---|
| `--split` | `all` | `validation`, `test` or `all` |
| `--terrains` | | pick terrains by name instead, e.g. `--terrains flat stairsUp` |
| `--difficulties` | `0.5 1.0` | difficulty levels to test |
| `--minSpeed`, `--maxSpeed` | `0.1`, `0.4` | range of forward speeds commanded (m/s) |
| `--numEnvs` | `512` | ducks per terrain. Lower it on a laptop |
| `--seconds` | `10` | episode length. Use 20 for the combined courses |
| `--seed` | `1000` | fixes the random terrain and spawns, so every checkpoint sees identical ground. Keep it different from training seeds |
| `--pushes` | off | keep random pushes on |
| `--domainRand` | off | randomize mass, friction etc. |
| `--out` | `results/eval.csv` | where the CSV goes |

The CSV has one row per duck: checkpoint, iteration, training seed, eval seed, terrain, tier, held out or not, difficulty, how far training got on that terrain type (`trainedUpTo`, `withinTraining`), pushes/randomization, commanded speed, whether it fell, mean velocity error and distance walked.

### Is "seen" really seen?

The curriculum decides which difficulty rows a duck actually trains on. Each checkpoint records, per training terrain type, the curriculum level the top 10% of its ducks reached (also logged as `terrain/reach90/<type>`). If you evaluate a `seen` terrain above that, `evaluate.py` prints a note and marks the rows `withinTraining = False`. Report those as extrapolation, not as seen.

### How many ducks is enough

A fall rate from n ducks has a standard error of √(p(1−p)/n), at most √(0.25/512) ≈ 2.2% with the default 512. So differences of a few percent between two runs aren't meaningful from one training seed. For the main comparisons, train each config with 3 seeds (`--seed 1`, `2`, `3`) and report the mean and spread across seeds.

### Eval terrains

At difficulty 1.0 (difficulty scales heights linearly from the minimum up to these maximums):

| terrain | tier | split | at difficulty 1.0 |
|---|---|---|---|
| flat | seen | validation | |
| bumpyGrid | seen | validation | random-height boxes up to 1 cm, 45 cm cells |
| smoothBumps | seen | validation | perlin noise up to 1.5 cm |
| slope | seen | validation | rise/run 0.10 (5.7°) |
| bumpyGridHard | harder | validation | boxes up to 2 cm |
| smoothBumpsHard | harder | validation | noise up to 3 cm |
| slopeHard | harder | validation | rise/run 0.18 (about 10°) |
| waves | novel | test | 1.0 cm amplitude, ~67 cm wavelength, max slope 0.094 (5.4°), inside the training slope range |
| wavesHard | novel | test | 1.5 cm amplitude, max slope 0.14 (8°): new shape and steeper |
| stairsDown | novel | test | spawns on top of a pyramid, 2 cm steps down |
| stairsUp | novel | test | spawns in a pit, 2 cm steps up |
| combinedSeen | combined | test | 2 m patchwork of the four training types |
| combinedAll | combined | test | 2 m patchwork of training types plus waves and both stairs |

The `harder` tier only goes past the training max above difficulty 0.5, so evaluate it at 1.0. The heights look tiny, but the duck only lifts its feet 1-2 cm, so these are real obstacles at its scale.

## Planned experiments

1. **Reward ablation.** Train `minimal`, `posture` and `full` on flat ground and compare gaits. `minimal` will probably find some weird shuffle or hop, which is the point.
2. **Terrain.** Train on `flat` vs `mixed`, then evaluate both on everything. This shows what the terrain curriculum buys on held-out terrain.
3. **Memory.** `--historyLength 1` vs `5` vs `10` (0.02 s, 0.1 s and 0.2 s of past at 50 Hz), all on mixed terrain. Pick N from the validation split, then compare on the test split. The prediction is that history helps most on terrain the duck has to feel out (bumpyGrid, stairsUp, combinedAll) and least on flat ground. If it doesn't, that's a real result too.
4. **Domain randomization.** On vs off, evaluated with `--pushes --domainRand` to see robustness.
5. **Baseline.** Compare against Pollen's trained rough-terrain policy.

Run each with a few seeds (`--seed`) so the comparisons hold up. Make every choice from validation results, then run the test split once at the end.

**Watch the curriculum on the first real run.** A duck only moves up a difficulty row if it ends an episode more than 4 m (half a tile) from where it spawned, and commands change direction every 3-8 s. If `env/Curriculum/terrain_levels/*` stay flat, the curriculum is the bottleneck, and the usual fixes are smaller training tiles or promoting on velocity-tracking error instead of distance.

## Why the versions are pinned

Don't bump these in `pyproject.toml` without testing first. Each pin fixes a real breakage:

- **`better-actuator-models` (BAM) at commit `62bd8ce`.** The BAM branch microduck_rl points at has moved on, and the newer version breaks microduck's actuator config (`unexpected keyword argument 'vin_drop_gain_range'`). It's pinned through `override-dependencies`, because a normal source entry conflicts with microduck_rl's own pin.
- **`mjviser==0.0.14` / `viser==1.0.30`.** Newer mjviser pulls in `mujoco-core` 4.x, which overwrites files from the `mujoco` 3.10 package the simulator needs (`libmujoco.so.3.10.0: cannot open shared object file`).
- **`mujoco`, `mujoco-warp`, `mjlab`, `warp-lang`, `torch`.** These match what microduck_rl was tested with.
- **The CPU torch index is limited to x86_64**, so it doesn't clash with microduck_rl's own torch rule for ARM machines.

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
| `KeyError` on a reward or observation name | Check the spelling against the tables printed at the start of the run. |
| tensorboard shows nothing | The run used `--logger wandb` or `none`. Only `--logger tensorboard` writes to `runs/`. |

## Working on this together

Ravi adds collaborators from the repo's Settings → Collaborators page. Accept the email invite, then clone as above.

Work on a branch and open a PR instead of pushing to `main`:

```bash
git checkout -b yourName/whatYouAreDoing
# ...changes...
git push -u origin yourName/whatYouAreDoing
gh pr create          # or open the PR on github.com
```

Don't commit `runs/`, `wandb/` or eval output, they're in `.gitignore`. Share checkpoints through wandb or a shared drive instead. If you change dependencies, commit the updated `uv.lock` in the same PR.
