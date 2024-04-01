from typing import List, Dict
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from models.blocks.core_blocks import Conv2dModule


class PytorchFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
    
    def forward(self, x):
        feats = {} # {"f0":x[0], "f1": x[1], "f2": x[2], "f3": x[3]}
        for i, feat in enumerate(x):
            feats["f" + str(i)] = x[i]
        return self.fpn(feats)


class FPN(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        no_norm_on_lateral: bool = False,
        norm: nn.Module = None,
        act: nn.Module = None,
        upsample_cfg=dict(mode='nearest')) -> None:
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        used_backbone_levels = len(in_channels)
        for i in range(used_backbone_levels):
            l_conv = Conv2dModule(
                in_channels[i],
                out_channels,
                1,
                norm=norm if not self.no_norm_on_lateral else None,
                act=act)
            fpn_conv = Conv2dModule(out_channels,
                                    out_channels,
                                    3,
                                    padding=1,
                                    norm=norm,
                                    act=act)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert len(inputs) == len(self.in_channels)

        x = list(inputs.values())
        # lateral path
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # output path
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        inputs.update(zip(inputs, outs))
        return inputs


class PAFPN(FPN):

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 no_norm_on_lateral: bool = False,
                 norm: nn.Module = None,
                 act: nn.Module = None) -> None:
        super(PAFPN, self).__init__(in_channels, out_channels,
                                    no_norm_on_lateral, norm, act)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        used_backbone_levels = len(in_channels)
        for _ in range(used_backbone_levels):
            d_conv = Conv2dModule(out_channels,
                                  out_channels,
                                  3,
                                  stride=2,
                                  padding=1,
                                  norm=norm,
                                  act=act)
            pafpn_conv = Conv2dModule(out_channels,
                                      out_channels,
                                      3,
                                      padding=1,
                                      norm=norm,
                                      act=act)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert len(inputs) == len(self.in_channels)

        x = list(inputs.values())
        # lateral paths
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # output paths
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        inputs.update(zip(inputs, outs))
        return inputs
