import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, sigmoid=True):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.sigmoid = sigmoid

    def forward(self, pred, target):
        if self.sigmoid:
            pred = pred.sigmoid()

        return self.mse_loss(pred, target)


class DiceLoss(nn.Module):
    def __init__(self, sigmoid=True, naive_dice=False, eps=1e-3):
        super(DiceLoss, self).__init__()
        self.sigmoid = sigmoid
        self.naive_dice = naive_dice
        self.eps = eps

    def forward(self, pred, target):
        if self.sigmoid:
            pred = pred.sigmoid()

        pred = pred.flatten(0)
        target = target.flatten(0).float()

        a = torch.sum(pred * target, dim=0)
        if self.naive_dice:
            b = torch.sum(pred, dim=0)
            c = torch.sum(target, dim=0)
            d = (2 * a + self.eps) / (b + c + self.eps)
        else:
            b = torch.sum(pred * pred, dim=0) + self.eps
            c = torch.sum(target * target, dim=0) + self.eps
            d = (2 * a) / (b + c)

        return 1 - d


if __name__ == "__main__":
    bad_pred = torch.zeros((160, 160)).float()
    bad_pred[0:40, 0:40] = 1.0
    good_pred = torch.zeros((160, 160)).float()
    good_pred[70:110, 60:100] = 1.0
    target = torch.zeros((160, 160)).float()
    target[60:100, 60:100] = 1.0
    print("Testing MSELoss")
    mse_loss = MSELoss(False)
    loss = mse_loss(bad_pred, target)
    print("Bad Prediction Loss: {l}".format(l=loss))
    loss = mse_loss(good_pred, target)
    print("Good Prediction Loss: {l}".format(l=loss))

    print("\nTesting DiceLoss\n")
    dice_loss = DiceLoss(False)
    loss = dice_loss(bad_pred, target)
    print("Bad Prediction Loss: {l}".format(l=loss))
    loss = dice_loss(good_pred, target)
    print("Good Prediction Loss: {l}".format(l=loss))
