from torch import nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork


class PytorchFPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels_list = cfg["in_channels_list"]
        out_channels = cfg["out_channels"]

        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
    
    def forward(self, x):
        feats = {}
        for i, feat in enumerate(x):
            feats["f" + str(i)] = feat
        return self.fpn(feats).values()


class FPN(nn.Module):

    def __init__(self, cfg):
        super(FPN, self).__init__()
        in_channels_list = cfg["in_channels_list"]
        out_channels = cfg["out_channels"]
        no_norm_on_lateral = cfg["no_norm_on_lateral"]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.num_inputs = len(in_channels_list)
        for i in range(self.num_inputs):
            l_conv = nn.Sequential(nn.Conv2d(in_channels_list[i], out_channels, 1))
            if not no_norm_on_lateral:
                l_conv.append(nn.BatchNorm2d(out_channels))
            l_conv.append(nn.ReLU())
            fpn_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU())

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, x):
        # lateral path
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # top-down path
        for i in range(self.num_inputs - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # output path
        outputs = [
            self.fpn_convs[i](laterals[i]) for i in range(self.num_inputs)
        ]
        return outputs


class PAFPN(FPN):

    def __init__(self, cfg):
        super(PAFPN, self).__init__(cfg)
        out_channels = cfg["out_channels"]
        
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for _ in range(self.num_inputs):
            downsample_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU())
            pafpn_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())
            self.downsample_convs.append(downsample_conv)
            self.pafpn_convs.append(pafpn_conv)

    def forward(self, x):
        # lateral paths
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # top-down path
        for i in range(self.num_inputs - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # output paths
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(self.num_inputs)
        ]

        # bottom-up path
        for i in range(0, self.num_inputs - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outputs = []
        outputs.append(inter_outs[0])
        outputs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, self.num_inputs)
        ])

        return outputs
