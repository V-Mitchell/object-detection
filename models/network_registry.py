from models.backbones.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.backbones.regnet import RegNetY600MF
from models.necks.fpn import PytorchFPN, FPN, PAFPN
from models.heads.detector_head import DetectorHead

BACKBONES = {
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152
}

NECKS = {
    "PytorchFPN": PytorchFPN,
    "FPN": FPN,
    "PAFPN": PAFPN
}

HEADS = {
    "DetectorHead" : DetectorHead
}

def get_backbone(cfg):
    return BACKBONES[cfg["network"]](3, cfg["out_channels_list"])

def get_neck(cfg):
    return NECKS[cfg["network"]](cfg)

def get_head(cfg):
    return HEADS[cfg["network"]](cfg)
