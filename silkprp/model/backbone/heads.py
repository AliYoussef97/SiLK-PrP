import torch.nn as nn
from silkprp.model.backbone.vgg_backbone import vgg_block


class Detector_Head(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        channels = config['channels']

        self.logits = nn.ModuleDict()
        self.logits['det_H1'] = vgg_block(channels[0], channels[1], 3, norm=config['normalization'], activation=config['activation'])
        self.logits['det_H2'] = vgg_block(channels[1], channels[-1], 1, norm=config['normalization'], activation=config['activation'], remove_activation=True)
                
    def forward(self, x):
        for block in self.logits.values():
            x = block(x)
        return x
        

class Descriptor_Head(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        channels = config['channels']

        self.raw_descriptors = nn.ModuleDict() 
        self.raw_descriptors['desc_H1'] = vgg_block(channels[0], channels[1], 3, norm=config['normalization'], activation=config['activation'])
        self.raw_descriptors['desc_H2'] = vgg_block(channels[1], channels[-1], 1, norm=config['normalization'], activation=config['activation'], remove_activation=True)
        
    def forward(self, x):
        for block in self.raw_descriptors.values():
            x = block(x)
        return x