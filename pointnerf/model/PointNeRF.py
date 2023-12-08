import torch.nn as nn
from pointnerf.model.backbone.vgg_backbone import VGG_Backbone
from pointnerf.model.backbone.heads import Detector_Head, Descriptor_Head

class Pointnerf(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.backbone = VGG_Backbone(config["backbone"])

        self.detector_head = Detector_Head(config["detector_head"])

        self.descriptor_head = Descriptor_Head(config["descriptor_head"])

    def forward(self, x):

        x = self.backbone(x)

        logits = self.detector_head(x)

        descriptors = self.descriptor_head(x)

        return {"logits": logits, "raw_descriptors": descriptors}