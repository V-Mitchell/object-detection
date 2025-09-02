from utils.dict import remove_key
from models.backbones.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.backbones.regnet import RegNetY600MF
from models.necks.fpn import FPN, PANet
from models.heads.dense_head import DenseHead, AnchorlessHead

BACKBONES = {
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152
}

NECKS = {"FPN": FPN, "PANet": PANet}

HEADS = {"DenseHead": DenseHead, "AnchorlessHead": AnchorlessHead}


def get_backbone(cfg):
    return BACKBONES[cfg["network"]]()


def get_neck(cfg):
    return NECKS[cfg["network"]](**remove_key(cfg, ["network"]))


def get_head(cfg):
    return HEADS[cfg["network"]](**remove_key(cfg, ["network"]))
