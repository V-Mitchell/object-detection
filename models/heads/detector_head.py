import torch
from torch import nn

class DetectorHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg["in_channels"]
        num_layers = cfg["num_layers"]
        self.num_classes = cfg["num_classes"]
        self.num_priors = cfg["num_priors"]

        cls_layers = []
        bbox_layers = []
        for i in range(num_layers):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            bbox_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))

        self.cls_layers = nn.Sequential(*cls_layers, nn.Conv2d(in_channels, self.num_priors * self.num_classes, 3, 1, 1))
        self.bbox_layers = nn.Sequential(*bbox_layers, nn.Conv2d(in_channels, self.num_priors * 4, 3, 1, 1))
    
    def forward(self, x):

        cls_preds = []
        bbox_preds = []
        for feat in x:
            cls_preds.append(self.cls_layers(feat))
            bbox_preds.append(self.bbox_layers(feat))
        
        return (cls_preds, bbox_preds)

    def decode_predictions(self, cls_preds, bbox_preds):
        cls_process = []
        bbox_process = []
        for cls_pred, bbox_pred in zip(cls_preds, bbox_preds):
            cls_process.append(cls_pred.squeeze().permute(1, 2, 0).reshape(-1, self.num_classes))
            bbox_process.append(bbox_pred.squeeze().permute(1, 2, 0).reshape(-1, 4))
        return {
            "cls_preds": torch.cat(cls_process),
            "bbox_preds": torch.cat(bbox_process)
        }
