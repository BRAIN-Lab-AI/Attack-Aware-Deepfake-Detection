# flake8: noqa
from __future__ import annotations
import math, os, sys, re, json, time, random, pathlib, logging
import numpy as np
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    # allow import without torch on docs/build
    torch = None
    nn = object
    F = None
    Dataset = object
    DataLoader = object
##############################################################################
# MODELS
##############################################################################


#Two-stream model (content + residual) + FPN mask head
class HighPassLayer(nn.Module):
    def __init__(self):
        super().__init__()
        k = np.zeros((5,3,3), dtype=np.float32)
        k[0] = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
        k[1] = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
        k[2] = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], np.float32)
        k[3] = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], np.float32)
        k[4] = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], np.float32)
        weight = np.zeros((5*3,3,3,3), dtype=np.float32)
        for i in range(5):
            for c in range(3):
                weight[i*3+c, c] = k[i]
        self.conv = nn.Conv2d(3, 15, 3, padding=1, bias=False)
        with torch.no_grad(): self.conv.weight.copy_(torch.from_numpy(weight))
        for p in self.parameters(): p.requires_grad = False
    def forward(self, x): return self.conv(x)

class ResidualAdapter(nn.Module):
    def __init__(self, in_ch=15, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(True),
            nn.Conv2d(width, width, 3, padding=1, stride=2),
            nn.BatchNorm2d(width), nn.ReLU(True),
            nn.Conv2d(width, width*2, 3, padding=1, stride=2),
            nn.BatchNorm2d(width*2), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class MaskHeadFPN(nn.Module):
    def __init__(self, c2=256, c3=512, c4=1024, c5=2048, r_ch=128):
        super().__init__()
        self.l5 = nn.Conv2d(c5,256,1); self.l4 = nn.Conv2d(c4,256,1)
        self.l3 = nn.Conv2d(c3,256,1); self.l2 = nn.Conv2d(c2,256,1)
        self.lr = nn.Conv2d(r_ch,256,1)
        self.out = nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.ReLU(True),
                                 nn.Conv2d(128,64,3,padding=1), nn.ReLU(True),
                                 nn.Conv2d(64,1,1))
    def forward(self, feats, rfeat):
        c2,c3,c4,c5 = feats
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        r  = self.lr(rfeat)
        p  = p2 + F.interpolate(r, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        m  = F.interpolate(p, scale_factor=4, mode="bilinear", align_corners=False)
        return self.out(m)

class UnFooledNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.l1, self.l2, self.l3, self.l4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.hp = HighPassLayer()
        self.res_adapter = ResidualAdapter(15,64)
        self.fc = nn.Sequential(nn.Linear(2048+128,256), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256,1))
        self.mask_head = MaskHeadFPN(256,512,1024,2048,128)
    def forward(self, x):
        x0 = self.stem(x)
        c2 = self.l1(x0); c3 = self.l2(c2); c4 = self.l3(c3); c5 = self.l4(c4)
        p = self.gap(c5).flatten(1)
        r = self.res_adapter(self.hp(x))
        rp = self.gap(r).flatten(1)
        logit = self.fc(torch.cat([p,rp],1)).squeeze(1)
        mask_logit = self.mask_head((c2,c3,c4,c5), r)
        return logit, mask_logit
