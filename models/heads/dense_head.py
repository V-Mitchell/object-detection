import torch
import torch.nn as nn
from models.heads.proto_net import ProtoNet
from models.assigner.assigner import TaskAlignedAssigner


class DenseHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_priors,
                 input_channels=256,
                 feat_channels=256,
                 stacked_convs=2,
                 num_prototypes=8,
                 bg_index=0,
                 image_size=[640, 640]) -> None:
        super(DenseHead, self).__init__()
        self.proto_head = ProtoNet(input_channels,
                                   feat_channels,
                                   stacked_convs=4,
                                   num_levels=5,
                                   num_prototypes=num_prototypes)
        self.assigner = TaskAlignedAssigner(num_classes)

        cls_convs = []
        reg_convs = []
        coeff_convs = []
        for i in range(stacked_convs):
            channels = input_channels if i == 0 else feat_channels
            cls_convs.append(nn.Conv2d(channels, feat_channels, kernel_size=3, stride=1,
                                       padding=1))
            reg_convs.append(nn.Conv2d(channels, feat_channels, kernel_size=3, stride=1,
                                       padding=1))
            coeff_convs.append(
                nn.Conv2d(channels, feat_channels, kernel_size=3, stride=1, padding=1))

        cls_convs.append(
            nn.Conv2d(feat_channels, num_priors * num_classes, kernel_size=3, stride=1, padding=1))
        reg_convs.append(
            nn.Conv2d(feat_channels, num_priors * 4, kernel_size=3, stride=1, padding=1))
        coeff_convs.append(
            nn.Conv2d(feat_channels,
                      num_priors * num_prototypes,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        # Normalize output
        cls_convs.append(nn.Sigmoid())
        reg_convs.append(nn.Sigmoid())
        coeff_convs.append(nn.Sigmoid())

        self.cls_layer = nn.Sequential(*cls_convs)
        self.reg_layer = nn.Sequential(*reg_convs)
        self.coeff_layer = nn.Sequential(*coeff_convs)
        self.num_classes = num_classes
        self.num_priors = num_priors
        self.num_prototypes = num_prototypes
        self.bg_index = bg_index
        self.image_size = image_size

    def forward(self, x):
        prototypes = self.proto_head(x)
        cls_preds = []
        bbox_preds = []
        coeff_preds = []
        for feat in x:
            cls_pred = self.cls_layer(feat)
            bbox_pred = self.reg_layer(feat)
            coeff_pred = self.coeff_layer(feat)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
            coeff_preds.append(coeff_pred)
        return (tuple(cls_preds), tuple(bbox_preds), tuple(coeff_preds), prototypes)

    def loss(self, preds, labels):
        cls_preds, bbox_preds, coeff_preds, prototypes = preds
        batch_size, _, _, _ = cls_preds[0].shape
        cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for cls_pred in cls_preds
        ], 1)
        bbox_preds = torch.cat(
            [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds],
            1)
        coeff_preds = torch.cat([
            coeff_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_prototypes)
            for coeff_pred in coeff_preds
        ], 1)
        cls_labels, bbox_labels, mask_labels = labels
        cls_losses = []
        bbox_losses = []
        mask_losses = []
        for cls_pred, bbox_pred, coeff_pred, cls_label, bbox_label, mask_label in zip(
                cls_preds, bbox_preds, coeff_preds, cls_labels, bbox_labels, mask_labels):
            print("Batch Cls Scores {shape}".format(shape=cls_pred.shape))
            print("Batch BBox Preds {shape}".format(shape=bbox_pred.shape))
            print("Batch Coeff Preds {shape}".format(shape=coeff_pred.shape))
            assigned_labels, assigned_cls, assigned_bboxes = self.assigner(
                cls_pred, bbox_pred, cls_label, bbox_label, self.bg_index)
            print("Assigned Labels {shape}".format(shape=assigned_labels.shape))
            print("Assigned Cls {shape}".format(shape=assigned_cls.shape))
            print("Assigned BBox {shape}".format(shape=assigned_bboxes.shape))


if __name__ == "__main__":
    import numpy as np

    batch_size = 1
    x0 = torch.Tensor(np.zeros((batch_size, 256, 160, 160)))
    x1 = torch.Tensor(np.zeros((batch_size, 256, 80, 80)))
    x2 = torch.Tensor(np.zeros((batch_size, 256, 40, 40)))
    x3 = torch.Tensor(np.zeros((batch_size, 256, 20, 20)))
    x4 = torch.Tensor(np.zeros((batch_size, 256, 10, 10)))
    print("Testing Dense Head with input shapes:")
    print("x0 {shape}".format(shape=x0.shape))
    print("x1 {shape}".format(shape=x1.shape))
    print("x2 {shape}".format(shape=x2.shape))
    print("x3 {shape}".format(shape=x3.shape))
    print("x4 {shape}\n".format(shape=x4.shape))
    dense_head = DenseHead(10, 1)
    input = (x0, x1, x2, x3, x4)
    output = dense_head(input)
    cls_preds, bbox_preds, coeff_preds, prototypes = output

    for i, x in enumerate(cls_preds):
        print("Class Scores x{num} Shape: {shape}".format(num=i, shape=x.shape))
    for i, x in enumerate(bbox_preds):
        print("BBox Predictions x{num} Shape: {shape}".format(num=i, shape=x.shape))
    for i, x in enumerate(coeff_preds):
        print("Mask Coeff Predictions x{num} Shape: {shape}".format(num=i, shape=x.shape))
    print("Prototypes Shape: {shape}".format(shape=prototypes.shape))

    # BG label = 0, FG labels > 0, num FG classes = num_classes - 1
    cls_gt = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 1, 2, 3]).to(torch.long).unsqueeze(dim=0)
    # BBox format: xyxy
    bbox_gt = torch.Tensor(
        np.array([[40, 40, 100, 100], [100, 40, 120, 60], [60, 40, 100, 100], [260, 260, 360, 360],
                  [260, 60, 300, 300], [260, 300, 360, 400], [500, 500, 600, 600],
                  [460, 200, 500, 500], [500, 560, 520, 600], [200, 460, 220,
                                                               600]])).float().unsqueeze(dim=0)
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

    loss = dense_head.loss(output, labels)
