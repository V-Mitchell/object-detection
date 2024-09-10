import torch
from torch import nn
import torch.nn.functional as F


class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256, highest_block=False):
        super(FPNBlock, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.highest_block = highest_block

    def forward(self, feats):
        x, y = feats
        x = self.conv0(x)
        if not self.highest_block:
            x += F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        return (x, self.conv1(x))


class FPN(nn.Module):
    def __init__(self, expansion, in_channels_list=[64, 128, 256, 512], out_channels=256):
        super(FPN, self).__init__()

        self.p0 = FPNBlock(in_channels_list[0] * expansion, out_channels)
        self.p1 = FPNBlock(in_channels_list[1] * expansion, out_channels)
        self.p2 = FPNBlock(in_channels_list[2] * expansion, out_channels)
        self.p3 = FPNBlock(in_channels_list[3] * expansion, out_channels, highest_block=True)
        self.p4 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, feats):
        x0, x1, x2, x3 = feats
        x, p3 = self.p3((x3, None))
        x, p2 = self.p2((x2, x))
        x, p1 = self.p1((x1, x))
        x, p0 = self.p0((x0, x))
        p4 = self.p4(p3)
        return (p0, p1, p2, p3, p4)


if __name__ == "__main__":
    import numpy as np

    expansion = 1
    x0 = torch.Tensor(np.zeros((1, 64, 160, 160)))
    x1 = torch.Tensor(np.zeros((1, 128, 80, 80)))
    x2 = torch.Tensor(np.zeros((1, 256, 40, 40)))
    x3 = torch.Tensor(np.zeros((1, 512, 20, 20)))
    print("Testing FPN with input shapes:")
    print("x0 {shape}".format(shape=x0.shape))
    print("x1 {shape}".format(shape=x1.shape))
    print("x2 {shape}".format(shape=x2.shape))
    print("x3 {shape}\n".format(shape=x3.shape))
    fpn = FPN(expansion)
    input = (x0, x1, x2, x3)
    output = fpn(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))

    expansion = 4
    x0 = torch.Tensor(np.zeros((1, 256, 160, 160)))
    x1 = torch.Tensor(np.zeros((1, 512, 80, 80)))
    x2 = torch.Tensor(np.zeros((1, 1024, 40, 40)))
    x3 = torch.Tensor(np.zeros((1, 2048, 20, 20)))
    print("\nTesting FPN with input shapes:")
    print("x0 {shape}".format(shape=x0.shape))
    print("x1 {shape}".format(shape=x1.shape))
    print("x2 {shape}".format(shape=x2.shape))
    print("x3 {shape}\n".format(shape=x3.shape))
    fpn = FPN(expansion)
    input = (x0, x1, x2, x3)
    output = fpn(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))
