import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from model.optimizers.ranger21 import Ranger21
from model.optimizers.sam import SAM


def get_optimizer(config: DictConfig, net: nn.Module) -> optim.Optimizer:
    if config.optimizer.name == "Adam":
        return optim.Adam(
            net.parameters(),
            lr=config.optimizer.adam.lr,
            weight_decay=config.optimizer.adam.weight_decay,
        )
    elif config.optimizer.name == "AdamW":
        return optim.AdamW(
            net.parameters(),
            lr=config.optimizer.adamW.lr,
            weight_decay=config.optimizer.adamW.weight_decay,
        )
    elif config.optimizer.name == "Ranger21":
        return Ranger21(
            net.parameters(),
            lr=config.optimizer.ranger21.lr,
            weight_decay=config.optimizer.ranger21.weight_decay,
            num_epochs=config.trainer.max_epochs,
            num_batches_per_epoch=config.optimizer.ranger21.num_batches_per_epoch,
        )
    elif config.optimizer.name == "SAM":
        if config.optimizer.sam.base_optimizer == "SGD":
            base_optimizer = optim.SGD()
            args = config.optimizer.sgd
        elif config.optimizer.sam.base_optimizer == "Adam":
            base_optimizer = optim.Adam()
            args = config.optimizer.adam
        elif config.optimizer.sam.base_optimizer == "AdamW":
            base_optimizer = optim.AdamW()
            args = config.optimizer.adamw
        else:
            raise ValueError(
                f"Unknown base optimizer of SAM: {config.optimizer.sam.base_optimizer}"
            )
        return SAM(
            net.parameters(),
            base_optimizer=base_optimizer,
            rho=config.optimizer.sam.rho,
            adaptive=config.optimizer.sam.adaptive,
            **args,
        )

    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")
