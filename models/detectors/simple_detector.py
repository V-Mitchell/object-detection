import yaml
import torch
from torch import nn
from models.network_registry import get_backbone, get_neck, get_head

class SimpleDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = get_backbone(cfg["backbone"])
        self.neck = get_neck(cfg["neck"])
        self.head = get_head(cfg["head"])

    def forward(self, x):
        feats = self.backbone(x) # returns tuple(fx)
        feats = self.neck(feats) # returns dict(fx)
        preds = self.head(feats) # returns dict(fx)
        return preds

if __name__ == "__main__":
    with open("./cfg/simple_detector.yaml") as stream:
        params = yaml.safe_load(stream)
    detector = SimpleDetector(params["model"])
    detector.train()

    x = torch.ones([1, 3, 600, 600])
    out = detector(x)
    cls_preds, bbox_preds = out
    print("Class Preds")
    for x in cls_preds:
        print(x.shape)
    print("BBox Preds")
    for x in bbox_preds:
        print(x.shape)