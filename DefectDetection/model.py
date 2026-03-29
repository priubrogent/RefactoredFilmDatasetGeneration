"""
Lightweight UNet for film defect detection.

Input:  9-channel tensor — [scan | restored_1 | restored_2], each BGR, float32 [0,1]
Output: 1-channel logit map — sigmoid → defect probability per pixel

Architecture:
  Encoder: conv blocks + max-pool, doubling features at each stage.
  Bottleneck: two conv layers at the deepest resolution.
  Decoder: transposed-conv upsampling + skip connections + conv blocks.
  Head: 1×1 conv → raw logit (no activation — use BCEWithLogitsLoss).

Param count at base_features=32, depth=4: ~7.7M — fits on a single GPU
and trains fast enough for iterative experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Args:
        in_channels:   Number of input channels (default 9: scan+r1+r2).
        base_features: Feature maps in the first encoder block.
                       Doubled at each encoder stage.
        depth:         Number of encoder/decoder stages (down-sample steps).
    """

    def __init__(self, in_channels: int = 9, base_features: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        enc_channels  = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(out_ch)
            ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        # --- Decoder ---
        self.ups     = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_ch  = base_features * (2 ** i)
            self.ups.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(out_ch + skip_ch, out_ch))
            ch = out_ch

        # --- Output head ---
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # Handle odd spatial sizes from max-pool
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)  # raw logits, shape: (B, 1, H, W)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> UNet:
    return UNet(
        in_channels=cfg.get("in_channels", 9),
        base_features=cfg.get("base_features", 32),
        depth=cfg.get("depth", 4),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = UNet(in_channels=9, base_features=32, depth=4)
    x   = torch.randn(2, 9, 256, 256)
    y   = net(x)
    print(f"Output shape : {y.shape}")           # (2, 1, 256, 256)
    print(f"Parameters   : {count_parameters(net):,}")
