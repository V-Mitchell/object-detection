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


def pair_wise_bce_loss(preds, targets):
    """
    preds: logits with shape [N, P, C] (after softmax/sigmoid)
    targets: class indices with shape [N, P, C] (each value is in [0, C - 1])
    """
    return F.binary_cross_entropy(preds, targets, reduction='none').sum(-1)


def pair_wise_focal_loss(preds, targets, gamma=2):
    """
    preds: logits with shape [P, C] (before softmax/sigmoid)
    targets: class indices with shape [P] (each value is in [0, C - 1])
    """
    pass


def metrics_topk(metrics, topk, eps=1e-9):
    _, num_priors = metrics.shape
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=True)
    topk_mask = (topk_metrics.max(dim=-1, keepdim=True)[0] > eps).tile(
        (1, topk))  # what if no metric is above eps?
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_priors).sum(dim=-2)
    is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)


def compute_max_iou_prior(ious):
    num_bbox_gt = ious.shape[-2]
    max_iou_idx = ious.argmax(dim=-2)  # gt label idx of which the prior has the max iou
    is_max_iou = F.one_hot(max_iou_idx, num_bbox_gt).permute(1, 0)
    return is_max_iou.to(ious.dtype)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, num_classes, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.num_classes = num_classes
        self.bg_idx = 0
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad
    def forward(self, cls_pred, bbox_pred, label_gt, bbox_gt):
        """
        Task Aligned Assigner:
            1. Compute allginment metrics between all bbox predictions and gt, based on IoU
            2. Select topk bboxes as candidates for each gt
            3. Assign gt label with max iou score per each prediction
            Note: - BG label is assumed to be 0, whereas all FG labels are > 0
            - BG predictions are represented as the absence of confidence in any FG predictions
              i.e. the total number of predicted classes is the number of FG classes

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
        bbox_gt_shape = bbox_gt.shape

        if bbox_gt_shape[0] == 0:
            assigned_gt_index = torch.full((num_priors, ), 0, device=device)
            assigned_labels = torch.full((num_priors, ), self.bg_idx, device=device)
            assigned_bboxes = torch.zeros((num_priors, 4), device=device)
            assigned_bboxes[:, 2:4] = 1.0
            assigned_cls = torch.zeros((num_priors, num_classes), device=device)
            return (assigned_gt_index, assigned_labels, assigned_cls, assigned_bboxes)

        # calculate IoUs between each gt bbox and each prediction bbox
        ious = calculate_ious(bbox_gt, bbox_pred)
        cls_pred = cls_pred.permute(1, 0)
        label_gt = label_gt.long()

        bbox_cls_scores = cls_pred[label_gt.squeeze(-1) - 1]
        # element-wise multiplication
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        is_in_topk = metrics_topk(alignment_metrics, self.topk, self.eps)

        mask_positive = is_in_topk
        mask_positive_sum = mask_positive.sum(dim=-2)  # sum of gt matches for each prior
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(0) > 1).repeat([bbox_gt_shape[0], 1])
            is_max_iou = compute_max_iou_prior(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)

        assigned_gt_index = mask_positive.argmax(dim=-2)
        assigned_labels = label_gt[assigned_gt_index]
        assigned_cls = F.one_hot(assigned_labels - 1, num_classes)
        assigned_bboxes = bbox_gt.reshape([-1, 4])[assigned_gt_index]
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels,
                                      torch.full_like(assigned_labels, self.bg_idx))

        return (assigned_gt_index, assigned_labels, assigned_cls, assigned_bboxes)


class SimOTAssigner(nn.Module):
    def __init__(self, num_classes, image_size, bbox_bias=3.0):
        super(SimOTAssigner, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.bbox_bias = bbox_bias

    def geometry_constraint(self, bbox_labels, grids):
        """
        bbox_labels: bbox labels of shape [N, 4]
        grids: anchor/prior centers of shape [P, 4] (format is [cx, cy, cx, cy])
        """
        cxcy_grids = grids[:, 0:2]
        x1, y1, x2, y2 = bbox_labels[:, 0], bbox_labels[:, 1], bbox_labels[:, 2], bbox_labels[:, 3]
        px = cxcy_grids[:, 0].unsqueeze(1)
        py = cxcy_grids[:, 1].unsqueeze(1)
        in_x = (px >= x1) & (px <= x2)
        in_y = (py >= y1) & (py <= y2)
        fg_mask = in_x & in_y
        fg_mask = fg_mask.permute(1, 0)
        del x1, y1, x2, y2, px, py, in_x, in_y

        # img = np.zeros((640, 640, 3), dtype=np.uint8)
        # for box in bbox_labels:
        #     x1, y1, x2, y2 = box.int().tolist()
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        # for point in cxcy_grids:
        #     x, y = point.int().tolist()
        #     cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=1)

        # cv2.imwrite("/home/lorrentz/git/object-detection/engine/deleteme/gridbbox.png", img)

        cxcy_labels = bbox_labels[:, 0:2]
        cxcy_labels[:, 0] += (bbox_labels[:, 2] - bbox_labels[:, 0]) / 2
        cxcy_labels[:, 1] += (bbox_labels[:, 3] - bbox_labels[:, 1]) / 2
        m, _ = cxcy_labels.shape
        n, _ = cxcy_grids.shape
        cxcy_labels = cxcy_labels.unsqueeze(0).repeat(n, 1, 1)
        cxcy_grids = cxcy_grids.unsqueeze(1).repeat(1, m, 1)
        dxdy_sqrd = (cxcy_labels - cxcy_grids)**2
        r_sqrd = dxdy_sqrd[:, :, 0] + dxdy_sqrd[:, :, 1]
        r_sqrd = r_sqrd.permute(1, 0)
        return fg_mask, r_sqrd

    def simota_matching(self, cost, pair_wise_ious, fg_mask):
        m, n = cost.shape  # m is number of gt, n is number of anchors
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(m):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # case that one anchor matches multiple gt
        if anchor_matching_gt.max() > 1:
            multi_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multi_match_mask], dim=0)
            matching_matrix[:, multi_match_mask] *= 0
            matching_matrix[cost_argmin, multi_match_mask] = 1

        # update the fg_mask with anchors that actually actually match to a gt
        matched_fg_mask = anchor_matching_gt > 0
        # fg_mask[fg_mask.clone()] = fg_mask_inboxes ??????

        matched_gt_idxs = matching_matrix[:, matched_fg_mask].argmax(0)
        return matched_gt_idxs, matched_fg_mask

    @torch.no_grad
    def forward(self, cls_preds, obj_preds, bbox_preds, grids, cls_labels, bbox_labels):
        """
        SimOT Assigner:

            P = number of priors/anchors
            C = number of classes
            N = number of ground truth labels
        Args:
            cls_pred (Tensor, float): class prediction logits, shape(P, C)
            bbox_pred (Tensor, float): bbox prediction logits, shape(P, 4)
            label_gt (Tensor, int): ground truth object labels, shape(N, 1)
            bbox_gt (Tensor, float): ground truth bbox labels, shape(N, 4)

        Returns:
            assigned_label (Tensor): (P)
            assigned_bbox (Tensor): (P, 4)
            assigned_cls (Tensor): (P, C)
        """
        n, _ = bbox_preds.shape  # number of anchors/priors/predictions
        device = bbox_preds.device
        # case for no ground truth labels
        if bbox_labels.shape[0] == 0:
            assigned_cls = torch.zeros((n, self.num_classes), dtype=torch.float, device=device)
            assigned_obj = torch.zeros((n, 1), dtype=torch.float, device=device)
            assigned_bbox = torch.tensor(-1, device=device)
            matched_gt_idxs = torch.tensor(-1, device=device)
            matched_fg_mask = torch.tensor(-1, device=device)
            return assigned_cls, assigned_obj, assigned_bbox, matched_gt_idxs, matched_fg_mask

        grids_ = torch.cat([grid.permute(1, 2, 0).reshape(-1, 4) for grid in grids])

        # unnormalize bbox labels
        if bbox_labels.dim() == 1:
            bbox_labels = bbox_labels.unsqueeze(0)
        bbox_labels[:, [0, 2]] *= self.image_size[0]
        bbox_labels[:, [1, 3]] *= self.image_size[1]
        fg_mask, _ = self.geometry_constraint(bbox_labels, grids_)
        del grids_

        m, _ = bbox_labels.shape  # number of ground truths

        # get pairwise class loss Ccls between jth prediction and ith ground truth
        # multiply objectness with cls predictions
        cls_preds_ = (F.sigmoid(cls_preds.float()) * F.sigmoid(obj_preds.float())).sqrt()
        cls_loss = pair_wise_bce_loss(
            cls_preds_.unsqueeze(0).repeat(m, 1, 1),
            F.one_hot(cls_labels.to(torch.long) - 1,
                      self.num_classes).float().unsqueeze(1).repeat(1, n,
                                                                    1))  # class labels span 1 - N
        # get pairwise reg loss Creg between jth prediction and ith ground truth
        bbox_loss = calculate_ious(bbox_labels, bbox_preds)
        # get pairwise center prior Ccp between eacj jth anchor and ith ground truth
        # get background class Cbg cost FocalLoss(Pcls, 0)
        # get foreground cost Cfg = Ccls + aCreg + Ccp
        # compute final cost matrix C by concatenating Cbg to Cfg of shape (m+1, n)
        cost = cls_loss + self.bbox_bias * bbox_loss + float(1e6) * (~fg_mask)
        matched_gt_idxs, matched_fg_mask = self.simota_matching(cost, bbox_loss, fg_mask)

        # two options:
        # 1. calculate loss over all predictions. don't think this would make sense since
        # you don't necessarily want all predicitons to match a ground truth

        # 2. calculate cls and bbox loss for matched_fg_mask,
        # obj loss over fg_mask using r2 as target.
        # i think this is the most coherent

        # assigned_cls = F.one_hot(cls_labels[matched_gt_idxs].to(torch.long) - 1, self.num_classes)
        assigned_cls = cls_labels[matched_gt_idxs].to(torch.long) - 1
        assigned_obj = torch.ones_like(obj_preds[matched_gt_idxs], dtype=torch.float)
        # assigned_obj = torch.zeros_like(obj_preds, dtype=torch.float)
        # assigned_obj[matched_fg_mask] = 1.0
        # assigned_obj = torch.where(matched_fg_mask, 1.0, 0.0).unsqueeze(-1).float()
        assigned_bbox = bbox_labels[matched_gt_idxs]

        return assigned_cls, assigned_obj, assigned_bbox, matched_gt_idxs, matched_fg_mask


class NaiveAssigner(nn.Module):
    def __init__(self, num_classes, image_size, bbox_bias=1.0):
        super(NaiveAssigner, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.bbox_bias = bbox_bias

    def geometry_constraint(self, bbox_labels, grids):
        """
        bbox_labels: bbox labels of shape [N, 4]
        grids: anchor/prior centers of shape [P, 4] (format is [cx, cy, cx, cy])
        """
        cxcy_grids = grids[:, 0:2]
        x1, y1, x2, y2 = bbox_labels[:, 0], bbox_labels[:, 1], bbox_labels[:, 2], bbox_labels[:, 3]
        px = cxcy_grids[:, 0].unsqueeze(1)
        py = cxcy_grids[:, 1].unsqueeze(1)
        in_x = (px >= x1) & (px <= x2)
        in_y = (py >= y1) & (py <= y2)
        fg_mask = in_x & in_y
        fg_mask = fg_mask.permute(1, 0)
        del x1, y1, x2, y2, px, py, in_x, in_y

        cxcy_labels = bbox_labels[:, 0:2]
        cxcy_labels[:, 0] += (bbox_labels[:, 2] - bbox_labels[:, 0]) / 2
        cxcy_labels[:, 1] += (bbox_labels[:, 3] - bbox_labels[:, 1]) / 2
        m, _ = cxcy_labels.shape
        n, _ = cxcy_grids.shape
        cxcy_labels = cxcy_labels.unsqueeze(0).repeat(n, 1, 1)
        cxcy_grids = cxcy_grids.unsqueeze(1).repeat(1, m, 1)
        dxdy_sqrd = (cxcy_labels - cxcy_grids)**2
        r_sqrd = dxdy_sqrd[:, :, 0] + dxdy_sqrd[:, :, 1]
        r_sqrd = r_sqrd.permute(1, 0)
        return fg_mask, r_sqrd

    @torch.no_grad
    def forward(self, cls_preds, obj_preds, bbox_preds, grids, cls_labels, bbox_labels):
        """
        """
        n, _ = bbox_preds.shape  # number of anchors/priors/predictions
        device = bbox_preds.device
        # case for no ground truth labels
        if bbox_labels.shape[0] == 0:
            assigned_cls = torch.zeros((n, self.num_classes), dtype=torch.float, device=device)
            assigned_obj = torch.zeros((n, 1), dtype=torch.float, device=device)
            assigned_bbox = torch.tensor(-1, device=device)
            matched_gt_idxs = torch.tensor(-1, device=device)
            fg_mask = torch.tensor(-1, device=device)
            return assigned_cls, assigned_obj, assigned_bbox, matched_gt_idxs, fg_mask

        grids_ = torch.cat([grid.permute(1, 2, 0).reshape(-1, 4) for grid in grids])

        # unnormalize bbox labels
        if bbox_labels.dim() == 1:
            bbox_labels = bbox_labels.unsqueeze(0)
        bbox_labels[:, [0, 2]] *= self.image_size[0]
        bbox_labels[:, [1, 3]] *= self.image_size[1]
        fg_mask, _ = self.geometry_constraint(bbox_labels, grids_)
        del grids_

        m, _ = bbox_labels.shape  # number of ground truths

        # get pairwise class loss Ccls between jth prediction and ith ground truth
        # multiply objectness with cls predictions
        cls_preds_ = (F.sigmoid(cls_preds.float()) * F.sigmoid(obj_preds.float())).sqrt()
        cls_loss = pair_wise_bce_loss(
            cls_preds_.unsqueeze(0).repeat(m, 1, 1),
            F.one_hot(cls_labels.to(torch.long) - 1,
                      self.num_classes).float().unsqueeze(1).repeat(1, n,
                                                                    1))  # class labels span 1 - N
        # get pairwise reg loss Creg between jth prediction and ith ground truth
        bbox_loss = calculate_ious(bbox_labels, bbox_preds)
        # get foreground cost Cfg = Ccls + aCreg + Ccp
        # compute final cost matrix C by concatenating Cbg to Cfg of shape (m+1, n)
        cost = cls_loss + self.bbox_bias * bbox_loss
        fg_mask = fg_mask.sum(0) > 0
        matched_gt_idxs = cost[:, fg_mask].argmax(0)

        assigned_cls = F.one_hot(cls_labels[matched_gt_idxs].to(torch.long) - 1, self.num_classes)
        assigned_obj = torch.ones_like(obj_preds[fg_mask], dtype=torch.float)
        assigned_bbox = bbox_labels[matched_gt_idxs]

        return assigned_cls, assigned_obj, assigned_bbox, matched_gt_idxs, fg_mask


if __name__ == "__main__":
    import numpy as np

    print("Testing Task Aligned Assigner:\n")

    num_priors = 10000
    num_classes = 8
    num_gt = 10
    # Normalize cls and bbox preds
    cls_pred = torch.randn((num_priors, num_classes)).softmax(dim=-1)
    bbox_pred = torch.randn((num_priors, 4)).sigmoid()

    # BG label = 0, FG labels > 0, num FG classes = num_classes
    # BG predictions are represented by an absence of FG prediction confidence
    label_gt = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 1, 2]).to(torch.long)
    # BBox format: xyxy
    bbox_gt = torch.Tensor(
        np.array([[10, 10, 100, 100], [100, 10, 120, 50], [50, 10, 100, 100], [250, 250, 350, 350],
                  [250, 50, 300, 300], [250, 300, 350, 250], [500, 500, 600, 600],
                  [450, 200, 500, 500], [500, 550, 520, 600], [200, 450, 220, 600]
                  ])).float() / 640.0
    print("Class Predictions Shape: {shape}".format(shape=cls_pred.shape))
    print("BBox Predictions Shape: {shape}".format(shape=bbox_pred.shape))
    print("GT Labels Shape: {shape}".format(shape=label_gt.shape))
    print("GT BBoxes Shape: {shape}\n\n".format(shape=bbox_gt.shape))

    task_aligned_assigner = TaskAlignedAssigner(num_classes)
    _, assigned_labels, assigned_cls, assigned_bboxes = task_aligned_assigner(
        cls_pred, bbox_pred, label_gt, bbox_gt)
    print("Positive Matches", (assigned_labels != task_aligned_assigner.bg_idx).sum())

    print("Assigned Labels Shape: {shape}".format(shape=assigned_labels.shape))
    print("Assigned BBoxes Shape: {shape}".format(shape=assigned_bboxes.shape))
    print("Assigned Classes Shape: {shape}\n\n".format(shape=assigned_cls.shape))
    pos_labels = 0
    for i, (assign_label, assign_bbox, assign_cls, pred_bbox, pred_cls) in enumerate(
            zip(assigned_labels, assigned_bboxes, assigned_cls, bbox_pred, cls_pred)):
        if assign_label != task_aligned_assigner.bg_idx:
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
