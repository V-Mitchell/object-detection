import torch
import torch.nn as nn
import torch.nn.functional as F
from models.heads.proto_net import ProtoNet
from models.assigner.assigner import TaskAlignedAssigner, SimOTAssigner
from models.loss.class_loss import CELoss, BCELoss
from models.loss.bbox_loss import CIoULoss, GIoULoss
from models.loss.mask_loss import DiceLoss


class AnchorlessHead(nn.Module):
    def __init__(self,
                 num_classes,
                 input_channels=[64, 128, 256, 512],
                 num_prototypes=8,
                 image_size=[640, 640]):
        super(AnchorlessHead, self).__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.image_size = image_size
        self.grids = None

        self.assigner = SimOTAssigner(num_classes, image_size)

        self.cls_loss = CELoss()
        self.obj_loss = BCELoss()
        self.bbox_loss = GIoULoss()
        self.mask_loss = DiceLoss(sigmoid=False)

        self.conv_cls = nn.ModuleList()
        self.conv_obj = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        self.conv_coeff = nn.ModuleList()
        for ch in input_channels:
            self.conv_cls.append(nn.Conv2d(ch, num_classes, kernel_size=3, stride=1, padding=1))
            self.conv_obj.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
            self.conv_reg.append(nn.Conv2d(ch, 4, kernel_size=3, stride=1, padding=1))
            self.conv_coeff.append(
                nn.Conv2d(ch, num_prototypes, kernel_size=3, stride=1, padding=1), )

    def forward(self, x):
        cls_preds = []
        obj_preds = []
        reg_preds = []
        coeff_preds = []
        for i, feat in enumerate(x):
            cls_preds.append(self.conv_cls[i](feat))
            obj_preds.append(self.conv_obj[i](feat))
            reg_preds.append(self.conv_reg[i](feat))
            coeff_preds.append(self.conv_coeff[i](feat))

        return (tuple(cls_preds), tuple(obj_preds), tuple(reg_preds), tuple(coeff_preds))

    def build_grids(self, bbox_preds):
        """
        build detection grid for each feature pyramid level in image space coords
        return: tuple grids for each feature pyramid level of (cx, cy, cx, cy) format
        """
        grids = []
        device = bbox_preds[0].device
        for bbox_pred in bbox_preds:
            _, _, row, col = bbox_pred.shape
            stride = self.image_size[0] / row
            grid = torch.zeros((4, row, col)).to(device)
            xidx = torch.arange(row).view(1, row).expand(row, col).to(device)
            yidx = torch.arange(col).view(col, 1).expand(row, col).to(device)
            grid[0] = xidx
            grid[1] = yidx
            grid[2] = xidx
            grid[3] = yidx
            grid = grid.float()
            grid += 0.5  # add 0.5 to shift center grid to the middle of the cells
            # feature space to image space encoding formula (s/2 + xs, s/2 + ys)
            grid *= stride
            # grid += (stride / 2) # why do this?
            grids.append(grid)
        return tuple(grids)

    def process_bbox(self, bbox_preds):
        """
        normalize and convert bbox encodings from [l,t,r,b] -> [x0,y0,x1,y1]
        [x - l, y - t, x + r, y + b]
        """
        if self.grids == None:
            self.grids = self.build_grids(bbox_preds)
        proc_bbox_preds = []
        for bbox_pred, grid in zip(bbox_preds, self.grids):
            bbox_pred = F.sigmoid(bbox_pred)
            bbox_pred = torch.cat([(bbox_pred[:, 0, :, :] * self.image_size[0]).unsqueeze(1),
                                   (bbox_pred[:, 1, :, :] * self.image_size[1]).unsqueeze(1),
                                   (bbox_pred[:, 2, :, :] * self.image_size[0]).unsqueeze(1),
                                   (bbox_pred[:, 3, :, :] * self.image_size[1]).unsqueeze(1)],
                                  dim=1)
            bbox_pred = torch.cat([bbox_pred[:, 0:2, :, :] * -1.0, bbox_pred[:, 2:, :, :]], dim=1)
            bbox_pred = bbox_pred + grid
            proc_bbox_preds.append(bbox_pred)
        return tuple(proc_bbox_preds)

    def reshape_preds(self, preds):
        cls_preds, obj_preds, bbox_preds, coeff_preds = preds
        batch_size, _, _, _ = cls_preds[0].shape
        cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for cls_pred in cls_preds
        ], 1)
        obj_preds = torch.cat(
            [obj_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for obj_pred in obj_preds], 1)
        bbox_preds = torch.cat(
            [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds],
            1)
        coeff_preds = torch.cat([
            coeff_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_prototypes)
            for coeff_pred in coeff_preds
        ], 1)
        return (cls_preds, obj_preds, bbox_preds, coeff_preds)

    def loss(self, preds, labels):
        cls_preds, obj_preds, bbox_preds, coeff_preds = preds
        bbox_preds = self.process_bbox(bbox_preds)
        cls_preds, obj_preds, bbox_preds, coeff_preds = self.reshape_preds(
            (cls_preds, obj_preds, bbox_preds, coeff_preds))
        device = cls_preds.device
        batches, _, _ = cls_preds.shape
        cls_labels, bbox_labels, mask_labels = labels
        cls_losses = []
        obj_losses = []
        bbox_losses = []
        mask_losses = []
        for batch in range(batches):
            cls_pred = cls_preds[batch]
            obj_pred = obj_preds[batch]
            bbox_pred = bbox_preds[batch]
            coeff_pred = coeff_preds[batch]
            cls_label = cls_labels[batch].to(device)
            bbox_label = bbox_labels[batch].to(device)
            mask_label = mask_labels[batch].to(device)
            # we want to assign each prediction with an optimal ground truth label or a background label
            # assignment strategy will assign the predictions with their optimal ground truths
            # class, bbox, and mask loss are calculated on the assignment result
            assigned_cls, assigned_obj, assigned_bbox, matched_gt_idxs, matched_fg_mask = self.assigner(
                cls_pred, obj_pred, bbox_pred, self.grids, cls_label, bbox_label)

            # we will only calculate the loss for positive matches between predictions and ground truths
            if matched_gt_idxs.dim() == 0:
                cls_losses.append(torch.zeros((1), dtype=torch.float, device=device))
                obj_losses.append(self.obj_loss(obj_pred, assigned_obj).unsqueeze(0))
                bbox_losses.append(torch.zeros((1), dtype=torch.float, device=device))
            else:
                cls_losses.append(
                    self.cls_loss(cls_pred[matched_fg_mask], assigned_cls.long()).unsqueeze(0))
                obj_losses.append(
                    self.obj_loss(obj_pred[matched_fg_mask], assigned_obj).unsqueeze(0))
                bbox_losses.append(self.bbox_loss(bbox_pred[matched_fg_mask], assigned_bbox))

        loss = dict()
        loss["class"] = torch.cat(cls_losses, dim=0).mean()
        loss["obj"] = torch.cat(obj_losses, dim=0).mean()
        loss["bbox"] = torch.cat(bbox_losses, dim=0).mean()
        # loss["mask"] = torch.cat(mask_losses, 0).mean()
        return loss


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
                                   num_levels=3,
                                   num_prototypes=num_prototypes)
        self.assigner = TaskAlignedAssigner(num_classes)
        self.cls_loss = BCELoss()
        self.bbox_loss = CIoULoss()
        self.mask_loss = DiceLoss(sigmoid=False)

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

    def reshape_preds(self, preds):
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
        return (cls_preds, bbox_preds, coeff_preds, prototypes)

    def loss(self, preds, labels):
        cls_preds, bbox_preds, coeff_preds, prototypes = self.reshape_preds(preds)
        device = cls_preds.device
        cls_labels, bbox_labels, mask_labels = labels
        cls_losses = []
        bbox_losses = []
        mask_losses = []
        for (cls_pred, bbox_pred, coeff_pred, prototype, cls_label, bbox_label,
             mask_label) in zip(cls_preds, bbox_preds, coeff_preds, prototypes, cls_labels,
                                bbox_labels, mask_labels):
            cls_label = cls_label.to(device)
            bbox_label = bbox_label.to(device)
            if mask_label.shape[0] > 0:
                mask_label = F.interpolate(mask_label.unsqueeze(0),
                                           scale_factor=0.25,
                                           mode="bilinear",
                                           align_corners=True).squeeze(dim=0)
            mask_label = mask_label.to(device)
            assigned_label_idx, assigned_label, assigned_cls, assigned_bbox = self.assigner(
                cls_pred, bbox_pred, cls_label, bbox_label)

            positive_mask = assigned_label != self.bg_index
            positive_mask_pred = self.proto_head.compute_masks(torch.permute(prototype, (1, 2, 0)),
                                                               coeff_pred[positive_mask]).permute(
                                                                   (2, 0, 1))
            positive_assigned_mask = mask_label[assigned_label_idx[positive_mask]]
            # try calculating the loss only for positive matches
            # different weighing of the loss between positive and negative predictions
            cls_losses.append(self.cls_loss(cls_pred, assigned_cls).unsqueeze(0))
            bbox_losses.append(self.bbox_loss(bbox_pred, assigned_bbox).unsqueeze(0))
            mask_losses.append(
                self.mask_loss(positive_mask_pred, positive_assigned_mask).unsqueeze(0))

        loss = dict()
        loss["class"] = torch.cat(cls_losses, 0).mean()
        loss["bbox"] = torch.cat(bbox_losses, 0).mean()
        loss["mask"] = torch.cat(mask_losses, 0).mean()
        return loss


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

    loss = dense_head.loss(output, labels)
    print("loss", loss)
