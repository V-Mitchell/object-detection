import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()
        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.mean()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction='none')
        return loss.mean()


if __name__ == "__main__":
    print("Testing Focal Loss")
    focal_loss = FocalLoss()

    num_preds = 10
    num_classes = 5
    input = torch.randn((num_preds, num_classes), requires_grad=True).sigmoid()
    target = torch.randint(0, num_classes, (1, num_preds)).flatten()
    target = F.one_hot(target, num_classes=num_classes)
    print("Input:", input, input.shape)
    print("Target:", target, target.shape)
    loss = focal_loss(input, target)
    loss.backward()
    print("Loss:", loss, loss.shape)

    print("\n\nTesting Binary Cross Entropy Loss")
    bce_loss = BCELoss()

    input = torch.randn((num_preds, num_classes), requires_grad=True).sigmoid()
    target = torch.randint(0, num_classes, (1, num_preds)).flatten()
    target = F.one_hot(target, num_classes=num_classes)
    print("Input:", input, input.shape)
    print("Target:", target, target.shape)
    loss = bce_loss(input, target)
    loss.backward()
    print("Loss:", loss, loss.shape)
