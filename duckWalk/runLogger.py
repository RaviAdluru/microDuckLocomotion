"""Thin wrapper so the trainer can log to wandb, tensorboard, or nothing with the same call.
"""

from dataclasses import asdict


class RunLogger:
    def __init__(self, backend, runDir, cfg):
        self.backend = backend
        self.writer = None
        if backend == "wandb":
            import wandb

            self.run = wandb.init(project=cfg.wandbProject, name=cfg.runName, config=asdict(cfg), dir=runDir)
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=runDir)
        elif backend != "none":
            raise ValueError(f"unknown logger '{backend}'")

    def log(self, values, step):
        if self.backend == "wandb":
            import wandb

            wandb.log(values, step=step)
        elif self.backend == "tensorboard":
            for key, value in values.items():
                self.writer.add_scalar(key, value, step)

    def close(self):
        if self.backend == "wandb":
            self.run.finish()
        elif self.backend == "tensorboard":
            self.writer.close()
