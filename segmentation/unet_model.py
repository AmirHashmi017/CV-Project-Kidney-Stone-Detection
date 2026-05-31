"""
U-Net Architecture for Kidney Stone Segmentation
Week 4 - CV Project
"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU blocks (the basic U-Net building block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net with 4 down-sampling and 4 up-sampling stages.
    Input  : (B, 3, H, W)  – RGB image
    Output : (B, 1, H, W)  – binary segmentation map (logits)
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs  = nn.ModuleList()
        self.ups    = nn.ModuleList()
        self.pool   = nn.MaxPool2d(2, 2)

        # ── Encoder ──────────────────────────────────
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # ── Bottleneck ───────────────────────────────
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Decoder ──────────────────────────────────
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))   # up-sample
            self.ups.append(DoubleConv(f * 2, f))                  # merge skip + up

        # ── Final 1×1 conv ───────────────────────────
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []

        # Encode
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]          # reverse for decoder

        # Decode
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)       # up-sample
            skip = skips[i // 2]

            # Handle odd input dimensions
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)  # double conv

        return self.final(x)
