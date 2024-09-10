import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_ious(bbox1, bbox2, eps=1e-9):
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    px1y1, px2y2 = bbox1[:, :, 0:2], bbox1[:, :, 2:4]
    gx1y1, gx2y2 = bbox2[:, :, 0:2], bbox2[:, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    intersection = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - intersection + eps
    return intersection / union


def metrics_topk(metrics, topk, largest=True, eps=1e-9):
    _, num_priors = metrics.shape
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    topk_mask = (topk_metrics.max(dim=-1, keepdim=True)[0] > eps).tile((1, topk))
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_priors).sum(dim=-2)
    is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)


def compute_max_iou_prior(ious):
    num_bbox_gt = ious.shape[-2]
    max_iou_idx = ious.argmax(dim=-2)
    is_max_iou = F.one_hot(max_iou_idx, num_bbox_gt).permute(1, 0)
    return is_max_iou.to(ious.dtype)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, num_classes, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad
    def forward(self, cls_pred, bbox_pred, label_gt, bbox_gt, bg_idx):
        """
        Task Aligned Assigner:
            1. Compute allginment between all bbox predictions and gt, based on IoU
            2. Select topk bboxes as candidates for each gt
            3. 
            4. 

            P = number of priors
            C = number of classes
            N = number of ground truth labels
        Args:
            cls_pred (Tensor, float): class prediction logits, shape(P, C)
            bbox_pred (Tensor, float): bbox prediction logits, shape(P, 4)
            label_gt (Tensor, int): ground truth bbox labels, shape(N, 1)
            bbox_gt (Tensor, float): ground        max_metrics_per_instance = alignment_metrics.max(axis=-1, keepdim=True)[0]

        Returns:
            assigned_label (Tensor): (P)
            assigned_bbox (Tensor): (P, 4)
            assigned_cls (Tensor): (P, C)
        """
        device = cls_pred.device
        num_priors, num_classes = cls_pred.shape
        num_bbox_gt, _ = bbox_gt.shape

        if num_bbox_gt == 0:
            assigned_labels = torch.full((num_priors), bg_idx)
            assigned_bboxes = torch.zeros((num_priors, 4))
            assigned_cls = torch.zeros((num_priors, num_classes))
            return (assigned_labels, assigned_bboxes, assigned_cls)

        # calculate IoUs between each gt bbox and each prediction bbox
        ious = calculate_ious(bbox_gt, bbox_pred)
        cls_pred = cls_pred.permute(1, 0)
        label_gt = label_gt.long()

        # bbox_cls_scores = torch.zeros((num_bbox_gt, num_priors), dtype=torch.float, device=device)
        bbox_cls_scores = cls_pred[label_gt.squeeze(-1)]
        # element-wise multiplication
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        is_in_topk = metrics_topk(alignment_metrics, self.topk)

        mask_positive = is_in_topk
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(0) > 1).repeat([num_bbox_gt, 1])
            is_max_iou = compute_max_iou_prior(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)

        assigned_gt_index = mask_positive.argmax(dim=-2)
        assigned_labels = label_gt[assigned_gt_index]
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels,
                                      torch.full_like(assigned_labels, bg_idx))
        assigned_bboxes = bbox_gt.reshape([-1, 4])[assigned_gt_index]
        assigned_cls = F.one_hot(assigned_labels, num_classes)

        return (assigned_labels, assigned_cls, assigned_bboxes)


class SimOTAAssigner(nn.Module):
    def __init__(self):
        super(SimOTAAssigner, self).__init__()
        pass

    @torch.no_grad
    def forward(self):
        pass


if __name__ == "__main__":
    import numpy as np

    print("Testing Task Aligned Assigner:\n")

    num_priors = 10000
    num_classes = 8
    num_gt = 10
    bg_idx = 0
    # Normalize cls and bbox preds
    cls_pred = torch.randn((num_priors, num_classes)).sigmoid()
    bbox_pred = torch.randn((num_priors, 4)).sigmoid() * 640

    # BG label = 0, FG labels > 0, num FG classes = num_classes - 1
    label_gt = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 1, 2, 3]).to(torch.long)
    # BBox format: xyxy
    bbox_gt = torch.Tensor(
        np.array([[10, 10, 100, 100], [100, 10, 120, 50], [50, 10, 100, 100], [250, 250, 350, 350],
                  [250, 50, 300, 300], [250, 300, 350, 250], [500, 500, 600, 600],
                  [450, 200, 500, 500], [500, 550, 520, 600], [200, 450, 220, 600]])).float()
    print("Class Predictions Shape: {shape}".format(shape=cls_pred.shape))
    print("BBox Predictions Shape: {shape}".format(shape=bbox_pred.shape))
    print("GT Labels Shape: {shape}".format(shape=label_gt.shape))
    print("GT BBoxes Shape: {shape}\n\n".format(shape=bbox_gt.shape))

    task_aligned_assigner = TaskAlignedAssigner(num_classes)
    assigned_labels, assigned_cls, assigned_bboxes = task_aligned_assigner(
        cls_pred, bbox_pred, label_gt, bbox_gt, bg_idx)

    print("Assigned Labels Shape: {shape}".format(shape=assigned_labels.shape))
    print("Assigned BBoxes Shape: {shape}".format(shape=assigned_bboxes.shape))
    print("Assigned Classes Shape: {shape}\n\n".format(shape=assigned_cls.shape))
    pos_labels = 0
    for i, (assign_label, assign_bbox, assign_cls, pred_bbox, pred_cls) in enumerate(
            zip(assigned_labels, assigned_bboxes, assigned_cls, bbox_pred, cls_pred)):
        if assign_label != bg_idx:
            pos_labels += 1
            print(
                "Label {l} Pred BBox {pbbox} Assign BBox {abbox} Pred Cls {pcls} Assign Cls {acls}\n"
                .format(l=assign_label,
                        pbbox=pred_bbox,
                        abbox=assign_bbox,
                        pcls=pred_cls,
                        acls=assign_cls))

        if pos_labels == 10:
            break
