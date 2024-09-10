import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self,
                 input_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 num_levels=3,
                 num_prototypes=8):
        super().__init__()
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * input_channels, input_channels, kernel_size=1)

        convs = []
        for i in range(stacked_convs):
            channels = input_channels if i == 0 else feat_channels
            convs.append(nn.Conv2d(channels, feat_channels, kernel_size=3, padding=1))
            convs.append(nn.BatchNorm2d(feat_channels))
            convs.append(nn.ReLU())

        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(feat_channels, num_prototypes, kernel_size=1)

    def forward(self, x):
        fusion = [x[0]]
        size = x[0].shape[-2:]
        for i in range(1, self.num_levels):
            f = F.interpolate(x[i], size=size, mode='bilinear')
            fusion.append(f)
        fusion = torch.cat(fusion, dim=1)
        fusion = self.fusion_conv(fusion)
        proto_feats = self.stacked_convs(fusion)
        return self.projection(proto_feats)

    def compute_masks(self, prototypes, coeffs):
        coeffs_trans = torch.transpose(coeffs, 0, 1)
        return torch.matmul(prototypes, coeffs_trans)


if __name__ == "__main__":
    import numpy as np

    batch_size = 1
    x0 = torch.Tensor(np.zeros((batch_size, 256, 160, 160)))
    x1 = torch.Tensor(np.zeros((batch_size, 256, 80, 80)))
    x2 = torch.Tensor(np.zeros((batch_size, 256, 40, 40)))
    x3 = torch.Tensor(np.zeros((batch_size, 256, 20, 20)))
    x4 = torch.Tensor(np.zeros((batch_size, 256, 10, 10)))
    print("Testing ProtoNet with input shapes:")
    print("x0 {shape}".format(shape=x0.shape))
    print("x1 {shape}".format(shape=x1.shape))
    print("x2 {shape}".format(shape=x2.shape))
    print("x3 {shape}".format(shape=x3.shape))
    print("x4 {shape}".format(shape=x4.shape))
    proto_net = ProtoNet(256, 256, stacked_convs=4, num_levels=5)

    input = (x0, x1, x2, x3, x4)
    prototypes = proto_net(input)
    print("Output Prototypes: {shape}\n".format(shape=prototypes.shape))
    prototypes = torch.squeeze(prototypes)
    prototypes = torch.permute(prototypes, (1, 2, 0))

    print("Testing Mask Computation")
    pred_coeffs = torch.randn((50, 8))
    print("Processed Prototypes: {shape}".format(shape=prototypes.shape))
    print("Prediction Coefficients: {shape}".format(shape=pred_coeffs.shape))

    masks = proto_net.compute_masks(prototypes, pred_coeffs)
    print("Masks: {shape}".format(shape=masks.shape))
