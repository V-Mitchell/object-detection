from torch import nn

class DetectorHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg["in_channels"]
        num_layers = cfg["num_layers"]
        num_classes = cfg["num_classes"]
        num_priors = cfg["num_priors"]

        cls_layers = []
        bbox_layers = []
        for i in range(num_layers):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            bbox_layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))

        self.cls_layers = nn.Sequential(*cls_layers, nn.Conv2d(in_channels, num_priors * num_classes, 3, 1, 1))
        self.bbox_layers = nn.Sequential(*bbox_layers, nn.Conv2d(in_channels, num_priors * 4, 3, 1, 1))
    
    def forward(self, x):

        cls_preds = []
        bbox_preds = []
        for feat in x:
            cls_preds.append(self.cls_layers(feat))
            bbox_preds.append(self.bbox_layers(feat))
        
        return (cls_preds, bbox_preds)

def compute_head_loss(preds, labels):
    pass
    # use cross entropy for class loss

    # loss for bbox
    
    # loss for mask