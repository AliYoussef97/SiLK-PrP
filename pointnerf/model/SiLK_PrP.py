import torch.nn as nn
from silkprp.model.backbone.vgg_backbone import VGG_Backbone
from silkprp.model.backbone.heads import Detector_Head, Descriptor_Head

class SiLKPrP(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.backbone = VGG_Backbone(config["backbone"])

        self.detector_head = Detector_Head(config["detector_head"])

        self.descriptor_head = Descriptor_Head(config["descriptor_head"])

    def forward(self, x):

        feature_map = self.backbone(x)

        logits = self.detector_head(feature_map)

        descriptors = self.descriptor_head(feature_map)

        return {"logits": logits, "raw_descriptors": descriptors}