from typing import Dict
from torch import nn
from torch import Tensor
from models.blocks.core_blocks import YBlock


def generate_stage(num: int, block_fun):
    return [block_fun() for _ in range(num)]


class RegNetY600MF(nn.Module):

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        group_w = 16
        self.stage0 = nn.Conv2d(in_channels, 32, 1)
        self.stage4 = YBlock(32, 48, 2, 1, group_w)
        self.stage8 = nn.Sequential(YBlock(48, 112, 2, 1, group_w),
                                    YBlock(112, 112, 1, 1, group_w),
                                    YBlock(112, 112, 1, 1, group_w))
        self.stage16 = nn.Sequential(
            YBlock(112, 256, 2, 1, group_w),
            *generate_stage(6, lambda: YBlock(256, 256, 1, 1, group_w)))
        self.stage32 = nn.Sequential(
            YBlock(256, 608, 1, 1, group_w),
            *generate_stage(3, lambda: YBlock(608, 608, 1, 2, group_w)))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x0 = self.stage0(x)
        x4 = self.stage4(x0)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x16 = self.stage32(x16)
        return {"f4": x4, "f8": x8, "f16": x16}
