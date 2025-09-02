import torch
from torch import nn
import torch.nn.functional as F


class PANet(nn.Module):
    def __init__(self, expansion, input_channels=[64, 128, 256, 512]):
        super(PANet, self).__init__()

        self.relu = nn.ReLU()
        self.p0_lat_conv = nn.Sequential(*[
            nn.Conv2d(input_channels[0] * expansion + input_channels[1] * expansion,
                      input_channels[0] * expansion,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(input_channels[0] * expansion), self.relu
        ])
        self.p1_lat_conv1 = nn.Sequential(*[
            nn.Conv2d(input_channels[1] * expansion + input_channels[2] * expansion,
                      input_channels[1] * expansion,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(input_channels[1] * expansion), self.relu
        ])
        self.p2_lat_conv1 = nn.Sequential(*[
            nn.Conv2d(input_channels[2] * expansion + input_channels[3] * expansion,
                      input_channels[2] * expansion,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(input_channels[2] * expansion), self.relu
        ])
        self.p1_lat_conv2 = nn.Sequential(*[
            nn.Conv2d(input_channels[1] * expansion * 2,
                      input_channels[1] * expansion,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(input_channels[1] * expansion), self.relu
        ])
        self.p2_lat_conv2 = nn.Sequential(*[
            nn.Conv2d(input_channels[2] * expansion * 2,
                      input_channels[2] * expansion,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(input_channels[2] * expansion), self.relu
        ])
        self.p3_lat_conv = nn.Sequential(*[
            nn.Conv2d(input_channels[3] * expansion * 2,
                      input_channels[3] * expansion,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(input_channels[3] * expansion), self.relu
        ])

        self.p0_down_conv = nn.Sequential(*[
            nn.Conv2d(input_channels[0] * expansion,
                      input_channels[1] * expansion,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(input_channels[1] * expansion), self.relu
        ])
        self.p1_down_conv = nn.Sequential(*[
            nn.Conv2d(input_channels[1] * expansion,
                      input_channels[2] * expansion,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(input_channels[2] * expansion), self.relu
        ])
        self.p2_down_conv = nn.Sequential(*[
            nn.Conv2d(input_channels[2] * expansion,
                      input_channels[3] * expansion,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(input_channels[3] * expansion), self.relu
        ])

    def forward(self, feats):
        x0, x1, x2, x3 = feats

        p2_lat = torch.cat(
            (x2, F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=True)), dim=1)
        p2_lat = self.p2_lat_conv1(p2_lat)
        p1_lat = torch.cat(
            (x1, F.interpolate(p2_lat, scale_factor=2, mode="bilinear", align_corners=True)),
            dim=1)
        p1_lat = self.p1_lat_conv1(p1_lat)
        p0_lat = torch.cat(
            (x0, F.interpolate(p1_lat, scale_factor=2, mode="bilinear", align_corners=True)),
            dim=1)
        p0 = self.p0_lat_conv(p0_lat)

        p1_lat = torch.cat((p1_lat, self.p0_down_conv(p0)), dim=1)
        p1 = self.p1_lat_conv2(p1_lat)
        p2_lat = torch.cat((p2_lat, self.p1_down_conv(p1)), dim=1)
        p2 = self.p2_lat_conv2(p2_lat)
        p3_lat = torch.cat((x3, self.p2_down_conv(p2)), dim=1)
        p3 = self.p3_lat_conv(p3_lat)

        return (p0, p1, p2, p3)


class FPN(nn.Module):
    def __init__(self, expansion, input_channels=[64, 128, 256, 512], out_channels=256):
        super(FPN, self).__init__()

        self.p4 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self.p3 = nn.Sequential(
            nn.Conv2d(input_channels[3] * expansion,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0))
        self.p2_lat = nn.Sequential(
            nn.Conv2d(input_channels[2] * expansion,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0))
        self.p1_lat = nn.Sequential(
            nn.Conv2d(input_channels[1] * expansion,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0))
        self.p0_lat = nn.Sequential(
            nn.Conv2d(input_channels[0] * expansion,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0))
        self.p2_smth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.p1_smth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.p0_smth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, feats):
        x0, x1, x2, x3 = feats
        p3 = self.p3(x3)
        p4 = self.p4(p3)
        p2 = self.upsample_add(p3, self.p2_lat(x2))
        p1 = self.upsample_add(p2, self.p1_lat(x1))
        p0 = self.upsample_add(p1, self.p0_lat(x0))
        p2 = self.p2_smth(p2)
        p1 = self.p2_smth(p1)
        p0 = self.p2_smth(p0)
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

    expansion = 1
    x0 = torch.Tensor(np.zeros((1, 64, 160, 160)))
    x1 = torch.Tensor(np.zeros((1, 128, 80, 80)))
    x2 = torch.Tensor(np.zeros((1, 256, 40, 40)))
    x3 = torch.Tensor(np.zeros((1, 512, 20, 20)))
    print("Testing PANet with input shapes:")
    print("x0 {shape}".format(shape=x0.shape))
    print("x1 {shape}".format(shape=x1.shape))
    print("x2 {shape}".format(shape=x2.shape))
    print("x3 {shape}\n".format(shape=x3.shape))
    fpn = PANet(expansion)
    input = (x0, x1, x2, x3)
    output = fpn(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))

    expansion = 4
    x0 = torch.Tensor(np.zeros((1, 256, 160, 160)))
    x1 = torch.Tensor(np.zeros((1, 512, 80, 80)))
    x2 = torch.Tensor(np.zeros((1, 1024, 40, 40)))
    x3 = torch.Tensor(np.zeros((1, 2048, 20, 20)))
    print("\nTesting PANet with input shapes:")
    print("x0 {shape}".format(shape=x0.shape))
    print("x1 {shape}".format(shape=x1.shape))
    print("x2 {shape}".format(shape=x2.shape))
    print("x3 {shape}\n".format(shape=x3.shape))
    fpn = PANet(expansion)
    input = (x0, x1, x2, x3)
    output = fpn(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))
