import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super(Bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = nn.Sequential()
        if first_block:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          padding=0), nn.BatchNorm2d(out_channels * self.expansion))

    def forward(self, x):
        y = self.relu(self.bn0(self.conv0(x)))
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))
        y += self.downsample(x)
        return self.relu(y)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super(BasicBlock, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.stride = stride
        self.downsample = nn.Sequential()
        if first_block and stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.relu(self.bn0(self.conv0(x)))
        y = self.bn1(self.conv1(y))
        y += self.downsample(x)
        return self.relu(y)


class ResNet(nn.Module):
    in_channels = 64

    def __init__(self,
                 ResBlock,
                 blocks_list,
                 out_channels_list=[64, 128, 256, 512],
                 num_channels=3):
        super(ResNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.in_channels), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer0 = self.create_layer(ResBlock,
                                        blocks_list[0],
                                        self.in_channels,
                                        out_channels_list[0],
                                        stride=1)
        self.layer1 = self.create_layer(ResBlock,
                                        blocks_list[1],
                                        out_channels_list[0] * ResBlock.expansion,
                                        out_channels_list[1],
                                        stride=2)
        self.layer2 = self.create_layer(ResBlock,
                                        blocks_list[2],
                                        out_channels_list[1] * ResBlock.expansion,
                                        out_channels_list[2],
                                        stride=2)
        self.layer3 = self.create_layer(ResBlock,
                                        blocks_list[3],
                                        out_channels_list[2] * ResBlock.expansion,
                                        out_channels_list[3],
                                        stride=2)

    def forward(self, x):
        x0 = self.layer0(self.conv0(x))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return (x0, x1, x2, x3)

    def create_layer(self, ResBlock, blocks, in_channels, out_channels, stride=1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(ResBlock(in_channels, out_channels, stride=stride, first_block=True))
            else:
                layers.append(ResBlock(out_channels * ResBlock.expansion, out_channels))

        return nn.Sequential(*layers)


def ResNet18(channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_channels=channels)


def ResNet34(channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_channels=channels)


def ResNet50(channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_channels=channels)


def ResNet101(channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_channels=channels)


def ResNet152(channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_channels=channels)


if __name__ == "__main__":
    import numpy as np

    input = torch.Tensor(np.zeros((1, 3, 640, 640)))
    print("Testing ResNet with input shape {shape}".format(shape=input.shape))

    print("\nResNet18:")
    resnet18 = ResNet18()
    output = resnet18(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))

    print("\nResNet34:")
    resnet34 = ResNet34()
    output = resnet34(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))

    print("\nResNet50:")
    resnet50 = ResNet50()
    output = resnet50(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))

    print("\nResNet101:")
    resnet101 = ResNet101()
    output = resnet101(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))

    print("\nResNet152:")
    resnet152 = ResNet152()
    output = resnet152(input)
    for i, x in enumerate(output):
        print("Feature{num} Shape: {shape}".format(num=i, shape=x.shape))
