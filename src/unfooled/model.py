from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HighPassLayer(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = np.zeros((5, 3, 3), dtype=np.float32)
        kernels[0] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        kernels[1] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        kernels[2] = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], np.float32)
        kernels[3] = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], np.float32)
        kernels[4] = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
        weight = np.zeros((5 * 3, 3, 3, 3), dtype=np.float32)
        for index in range(5):
            for channel in range(3):
                weight[index * 3 + channel, channel] = kernels[index]
        self.conv = nn.Conv2d(3, 15, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(weight))
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class ResidualAdapter(nn.Module):
    def __init__(self, in_ch: int = 15, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, width, 3, padding=1, stride=2),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, width * 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MaskHeadFPN(nn.Module):
    def __init__(
        self,
        c2: int = 256,
        c3: int = 512,
        c4: int = 1024,
        c5: int = 2048,
        r_ch: int = 128,
    ):
        super().__init__()
        self.l5 = nn.Conv2d(c5, 256, 1)
        self.l4 = nn.Conv2d(c4, 256, 1)
        self.l3 = nn.Conv2d(c3, 256, 1)
        self.l2 = nn.Conv2d(c2, 256, 1)
        self.lr = nn.Conv2d(r_ch, 256, 1)
        self.out = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features, residual_feature: torch.Tensor) -> torch.Tensor:
        c2, c3, c4, c5 = features
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = self.l3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = self.l2(c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode="bilinear", align_corners=False
        )
        residual = self.lr(residual_feature)
        pyramid = p2 + F.interpolate(
            residual, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        mask = F.interpolate(
            pyramid, scale_factor=4, mode="bilinear", align_corners=False
        )
        return self.out(mask)


class UnFooledNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.l1 = backbone.layer1
        self.l2 = backbone.layer2
        self.l3 = backbone.layer3
        self.l4 = backbone.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.hp = HighPassLayer()
        self.res_adapter = ResidualAdapter(15, 64)
        self.fc = nn.Sequential(
            nn.Linear(2048 + 128, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.mask_head = MaskHeadFPN(256, 512, 1024, 2048, 128)

    def forward(self, inputs: torch.Tensor):
        stem = self.stem(inputs)
        c2 = self.l1(stem)
        c3 = self.l2(c2)
        c4 = self.l3(c3)
        c5 = self.l4(c4)
        content = self.gap(c5).flatten(1)
        residual = self.res_adapter(self.hp(inputs))
        pooled_residual = self.gap(residual).flatten(1)
        logit = self.fc(torch.cat([content, pooled_residual], 1)).squeeze(1)
        mask_logit = self.mask_head((c2, c3, c4, c5), residual)
        return logit, mask_logit
