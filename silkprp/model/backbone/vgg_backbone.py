import torch.nn as nn
from typing import Iterable


def assertion(input: str, *args) -> nn.Module:
    if input.lower() != "relu" and input.lower() != "batchnorm2d":
        raise NotImplementedError("Only ReLU and BatchNorm2d are supported")
    else:
        return nn.ReLU(inplace=True) if input == "ReLU" else nn.BatchNorm2d(*args, affine=True)

def vgg_block(input_dim: int,
              output_dim: int,
              kernel_size: int,
              norm: str = "BatchNorm2d",
              activation: str = "ReLU",
              remove_activation: bool = False,
              padding: int = 0) -> nn.Module:

    vgg_blk = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
                            assertion(activation),
                            assertion(norm, output_dim))
    
    if remove_activation: vgg_blk.__delitem__(1)
    
    return vgg_blk


class VGG_Backbone(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        assert isinstance(config["channels"], Iterable), "channels must be an iterable"
        channels = config["channels"]

        self.shared_backbone = nn.ModuleDict()

        for i in range(len(channels)-1):
            block = nn.Sequential(vgg_block(channels[i], channels[i+1], 3, norm=config["normalization"], activation=config["activation"]),
                                  vgg_block(channels[i+1], channels[i+1], 3, norm=config["normalization"], activation=config["activation"]))
            self.shared_backbone[f'block_{i}'] = block

    def forward(self, x):
        for block in self.shared_backbone.values():
            x = block(x)
        return x