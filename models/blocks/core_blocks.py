from typing import List
from torch import nn
from torch import Tensor
from models.blocks.conv import Conv2dModule
from models.blocks.modules import SE2dModule, DilatedConvCatModule


class XBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 group_w: int, bottleneck_ratio: float,
                 se_ratio: float) -> None:
        super(XBlock, self).__init__()
        base_channels = int(round(out_channels * bottleneck_ratio))
        self.convbnrelu1 = Conv2dModule(in_channels,
                                        base_channels,
                                        kernel_size=1,
                                        bias=False,
                                        norm=nn.BatchNorm2d(base_channels),
                                        act=nn.ReLU(inplace=True))
        n_groups = base_channels // group_w
        self.convbnrelu2 = Conv2dModule(base_channels,
                                        base_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        groups=n_groups,
                                        bias=False,
                                        norm=nn.BatchNorm2d(base_channels),
                                        act=nn.ReLU(inplace=True))
        self.with_se = se_ratio > 0
        if self.with_se:
            se_channels = int(round(in_channels * se_ratio))
            self.se = SE2dModule(base_channels, se_channels)

        self.convbn3 = Conv2dModule(base_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    norm=nn.BatchNorm2d(out_channels))
        self.act3 = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2dModule(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=stride,
                                         bias=False,
                                         norm=nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        if self.with_se:
            x = self.se(x)
        x = self.convbn3(x)
        return self.act3(x + shortcut)


class YBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 dilation: int, group_w: int) -> None:
        super(YBlock, self).__init__()
        groups = out_channels // group_w
        self.convbnrelu1 = Conv2dModule(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        bias=False,
                                        norm=nn.BatchNorm2d(out_channels),
                                        act=nn.ReLU(inplace=True))
        self.convbnrelu2 = Conv2dModule(out_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=dilation,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=False,
                                        norm=nn.BatchNorm2d(out_channels),
                                        act=nn.ReLU(inplace=True))
        self.convbn3 = Conv2dModule(out_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    norm=nn.BatchNorm2d(out_channels))
        self.act3 = nn.ReLU(inplace=True)
        self.se = SE2dModule(out_channels, in_channels // 4)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2, ceil_mode=True),
                Conv2dModule(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=stride,
                             bias=False,
                             norm=nn.BatchNorm2d(out_channels)))
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        x = self.se(x)
        x = self.convbn3(x)
        return self.act3(x + shortcut)


class DBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 dilations: List[int],
                 group_w,
                 attention: str = "se") -> None:
        super(DBlock, self).__init__()
        groups = out_channels // group_w
        self.convbnrelu1 = Conv2dModule(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        bias=False,
                                        norm=nn.BatchNorm2d(out_channels),
                                        act=nn.ReLU(inplace=True))
        if len(dilations) == 1:
            dilation = dilations[0]
            conv2 = nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              groups=groups,
                              padding=dilation,
                              dilation=dilation,
                              bias=False)
        else:
            conv2 = DilatedConvCatModule(out_channels,
                                         dilations,
                                         group_w=group_w,
                                         stride=stride,
                                         bias=False)

        self.convbnrelu2 = nn.Sequential(conv2, nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))
        self.convbn3 = Conv2dModule(out_channels,
                                    out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    norm=nn.BatchNorm2d(out_channels))
        self.act3 = nn.ReLU(inplace=True)
        if attention == "se":
            self.se = SE2dModule(out_channels, in_channels // 4)
        elif attention == "se2":
            self.se = SE2dModule(out_channels, out_channels // 4)
        else:
            self.se = None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2, ceil_mode=True),
                Conv2dModule(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=stride,
                             bias=False,
                             norm=nn.BatchNorm2d(out_channels)))
        else:
            self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.convbnrelu1(x)
        x = self.convbnrelu2(x)
        if self.se:
            x = self.se(x)
        x = self.convbn3(x)
        return self.act3(x + shortcut)
