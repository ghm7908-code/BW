import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import convrelu


class RoofStructurePrior(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.proj_layer2 = convrelu(512, 256, 1, 0)
        self.conv_up1 = convrelu(256 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(256 + 64, hidden_dim, 3, 1)
        self.conv_original = convrelu(hidden_dim + 64, hidden_dim, 3, 1)
        self.output = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, conv_outputs):
        x = self.proj_layer2(conv_outputs['layer2'])
        x = self.upsample(x)
        x = self.conv_up1(self._concat(x, conv_outputs['layer1']))

        x = self.upsample(x)
        x = self.conv_up0(self._concat(x, conv_outputs['layer0']))

        x = self.upsample(x)
        x = self.conv_original(self._concat(x, conv_outputs['x_original']))

        roof_features = x
        pred = self.output(x).squeeze(1)
        return pred, roof_features

    @staticmethod
    def _concat(x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        return torch.cat([x, skip], dim=1)
