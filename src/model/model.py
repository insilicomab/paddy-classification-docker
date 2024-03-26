import timm
import torch.nn as nn
from omegaconf import DictConfig


class TimmNet(nn.Module):
    def __init__(self, config: DictConfig):
        super(TimmNet, self).__init__()
        self.config = config
        self.model_name = config.net.model_name
        self.num_classes = config.num_classes
        self.pretrained = config.net.pretrained
        self.net = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
        )

    def forward(self, x):
        return self.net(x)


def get_model(config: DictConfig):
    return TimmNet(config)
