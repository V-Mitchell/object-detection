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
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)
    
    def loss(self, preds, labels):
        return self.head.loss(preds, labels)

if __name__ == "__main__":
    print("Testing SimpleDetector")
    with open("./cfg/simple_detector.yaml") as stream:
        params = yaml.safe_load(stream)
    print(params)
    detector = SimpleDetector(params["model"])
    detector.train()

    x = torch.zeros([1, 3, 640, 640])
    output = detector(x)
    cls_preds, bbox_preds, coeff_preds, prototypes = output

    for i, x in enumerate(cls_preds):
        print("Class Scores x{num} Shape: {shape}".format(num=i, shape=x.shape))
    for i, x in enumerate(bbox_preds):
        print("BBox Predictions x{num} Shape: {shape}".format(num=i, shape=x.shape))
    for i, x in enumerate(coeff_preds):
        print("Mask Coeff Predictions x{num} Shape: {shape}".format(num=i, shape=x.shape))
    print("Prototypes Shape: {shape}".format(shape=prototypes.shape))
