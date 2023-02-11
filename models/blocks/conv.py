import torch.nn as nn
from torch import Tensor


class Conv2dModule(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm: nn.Module = None,
                 act: nn.Module = None) -> None:
        super(Conv2dModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation, groups, bias))
        if norm is not None:
            self.conv.append(norm)
        if act is not None:
            self.conv.append(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)