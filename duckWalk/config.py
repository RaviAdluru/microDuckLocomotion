"""All the knobs for a training run in one dataclass.
train.py turns every field into a command line flag, e.g. --numEnvs 1024 --rewardSet minimal.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    runName: str = "baseline"
    seed: int = 1
    device: str = "cuda:0"

    # environment
    numEnvs: int = 4096
    rewardSet: str = "full"  # minimal | posture | full  (see rewards.py)
    terrainSet: str = "mixed"  # flat | mixed | mixedUniform  (see terrains.py)
    historyLength: int = 1  # frames of actor obs stacked together, 1 = no memory
    domainRand: bool = True  # mass/friction/motor gain randomization + pushes
    # start the action-rate penalty small and tighten it over training (Pollen's schedule),
    # so the policy can explore big leg motions before it is pushed toward smooth ones
    actionRateRamp: bool = False
    trackLinWeight: float = 2.0  # weight on the forward/sideways speed tracking reward

    # networks
    hiddenDims: tuple[int, ...] = field(default_factory=lambda: (512, 256, 128))
    initStd: float = 1.0

    # PPO
    maxIterations: int = 3000
    numStepsPerEnv: int = 24
    learningRate: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clipRatio: float = 0.2
    valueCoef: float = 1.0
    entropyCoef: float = 0.01
    numEpochs: int = 5
    numMiniBatches: int = 4
    targetKl: float = 0.01  # lr is adapted to keep KL near this
    maxGradNorm: float = 1.0

    # logging / checkpoints
    logger: str = "wandb"  # wandb | tensorboard | none
    wandbProject: str = "microDuckLocomotion"
    saveInterval: int = 250
    resume: str = ""  # path to a checkpoint to keep training from
