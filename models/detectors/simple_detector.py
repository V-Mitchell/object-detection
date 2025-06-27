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
    import yaml
    import numpy as np
    print("Testing SimpleDetector")
    with open("./cfg/simple_detector.yaml") as stream:
        params = yaml.safe_load(stream)
    print(params)
    model = SimpleDetector(params["model"])
    model.train()

    x = torch.zeros([1, 3, 640, 640])
    output = model(x)
    cls_preds, bbox_preds, coeff_preds, prototypes = output

    for i, x in enumerate(cls_preds):
        print("Class Scores x{num} Shape: {shape}".format(num=i, shape=x.shape))
    for i, x in enumerate(bbox_preds):
        print("BBox Predictions x{num} Shape: {shape}".format(num=i, shape=x.shape))
    for i, x in enumerate(coeff_preds):
        print("Mask Coeff Predictions x{num} Shape: {shape}".format(num=i, shape=x.shape))
    print("Prototypes Shape: {shape}".format(shape=prototypes.shape))

    # BG label = 0, FG labels > 0, num FG classes = num_classes - 1
    cls_gt = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 1, 2, 3]).to(torch.int32).unsqueeze(dim=0)
    # BBox format: xyxy
    bbox_gt = torch.Tensor(
        np.array([[40, 40, 100, 100], [100, 40, 120, 60], [60, 40, 100, 100], [260, 260, 360, 360],
                  [260, 60, 300, 300], [260, 300, 360, 400], [500, 500, 600, 600],
                  [460, 200, 500, 500], [500, 560, 520, 600], [200, 460, 220, 600]
                  ])).float().unsqueeze(dim=0) / 640.0
    mask_gt = torch.zeros((10, 160, 160), dtype=torch.float)
    mask_gt[0, 10:25, 10:25] = 1.0
    mask_gt[1, 25:30, 10:15] = 1.0
    mask_gt[2, 15:25, 10:25] = 1.0
    mask_gt[3, 65:90, 65:90] = 1.0
    mask_gt[4, 65:75, 15:75] = 1.0
    mask_gt[5, 65:90, 75:100] = 1.0
    mask_gt[6, 125:150, 125:150] = 1.0
    mask_gt[7, 115:125, 50:125] = 1.0
    mask_gt[8, 125:130, 140:150] = 1.0
    mask_gt[9, 50:55, 115:150] = 1.0
    mask_gt = mask_gt.unsqueeze(dim=0)
    print("GT Labels Shape: {shape}".format(shape=cls_gt.shape))
    print("GT BBoxes Shape: {shape}".format(shape=bbox_gt.shape))
    print("GT Masks Shape: {shape}\n".format(shape=mask_gt.shape))
    labels = (cls_gt, bbox_gt, mask_gt)

    loss = model.loss(output, labels)
