from typing import List
import torch
from torch import nn
from torch import Tensor


class SE2dModule(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(SE2dModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


class DilatedConvCatModule(nn.Module):

    def __init__(self, channels: int, dilations: List[int], group_w: int,
                 stride: int, bias: bool) -> None:
        super(DilatedConvCatModule, self).__init__()
        n_splits = len(dilations)
        assert channels % n_splits == 0, "Number for channels must be divisible by number of dialations"
        sub_channels = channels // n_splits
        assert sub_channels % group_w == 0, "Number of sub-channels must be divisible by group width"
        groups = sub_channels // group_w
        convs = []
        for d in dilations:
            convs.append(
                nn.Conv2d(sub_channels,
                          sub_channels,
                          3,
                          padding=d,
                          stride=stride,
                          bias=bias,
                          groups=groups))
        self.convs = nn.ModuleList(convs)
        self.n_splits = n_splits

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tensor_split(x, self.n_splits, dim=1)
        res = [self.convs[i](x[i]) for i in range(self.n_splits)]
        return torch.cat(res, dim=1)