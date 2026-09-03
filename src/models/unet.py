"""Small, easy-to-read U-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''
        两次卷积
        第一次卷积：提取局部特征
        第二次卷积：特征整合
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        in_ch = 3  # 输入通道数RGB
        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        in_ch = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_ch, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
            in_ch = f
        self.head = nn.Conv2d(features[0], 1, 1)
        # 输出通道数1 得到[B, 1, H, W] 模型输出logits 预测时进行sigmoid

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return self.head(x)
