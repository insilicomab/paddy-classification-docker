import torch.optim as optim
from omegaconf import DictConfig


def get_scheduler(config: DictConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler:
    if config.scheduler.name == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.CosineAnnealingWarmRestarts.T_0,
            eta_min=config.scheduler.CosineAnnealingWarmRestarts.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler.name}")
