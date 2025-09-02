import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target, weights=None):
        """
        Input: Class logits of shape [N, C] (input is unnormalized)
        Target: Target of shape [N, C]
        """
        target = target.to(torch.int64)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        pt = logpt.data.exp()
        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.mean()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target, weights=None):
        """
        Input: Class logits of shape [N, C] (input is unnormalized)
        Target: Target of shape [N, C]
        """
        loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction='mean')
        return loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, input, target, weights=None):
        """
        Input: Class logits of shape [N, C] (input is unnormalized)
        Target: Target of shape [N]
        """
        loss = F.cross_entropy(input, target, reduction="mean")
        return loss


if __name__ == "__main__":
    print("Testing Focal Loss")
    focal_loss = FocalLoss()

    num_preds = 10
    num_classes = 5
    input = torch.randn((num_preds, num_classes), requires_grad=True)
    target = torch.randint(0, num_classes, (1, num_preds)).flatten()
    target = F.one_hot(target, num_classes=num_classes)
    print("Input:", input, input.shape)
    print("Target:", target, target.shape)
    loss = focal_loss(input, target)
    loss.backward()
    print("Loss:", loss, loss.shape)

    print("\n\nTesting Cross Entropy Loss")
    ce_loss = CELoss()

    input = torch.randn((num_preds, num_classes), requires_grad=True)
    target = torch.randint(0, num_classes, (1, num_preds)).flatten()
    print("Input:", input, input.shape)
    print("Target:", target, target.shape)
    loss = ce_loss(input, target)
    loss.backward()
    print("Loss:", loss, loss.shape)
