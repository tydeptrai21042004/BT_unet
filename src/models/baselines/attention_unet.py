from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..common.blocks import DoubleConv
from ..common.encoder import PyramidEncoder
from ..common.utils import ensure_tuple_channels, init_weights, resize_to
from ..registry import register_model


class AttentionGate(nn.Module):
    """Additive attention gate used by Attention U-Net.

    This gate follows the paper-level Attention U-Net mechanism: project the
    skip tensor and decoder gating tensor to an intermediate space, add them,
    pass the result through ReLU + sigmoid, and use the resulting spatial gate
    to filter the skip connection before concatenation.
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        inter_channels = max(int(inter_channels or skip_channels // 2), 1)
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = resize_to(gate, skip)
        attn = self.psi(self.W_x(skip) + self.W_g(gate))
        return skip * attn


class AttentionUNetDecoderBlock(nn.Module):
    """Attention U-Net decoder block with an additive attention-gated skip."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.attention = AttentionGate(
            skip_channels=skip_channels,
            gate_channels=in_channels,
            inter_channels=max(skip_channels // 2, 1),
        )
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, norm=norm, act=act)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = resize_to(x, skip)
        skip = self.attention(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))


@register_model("attention_unet")
class AttentionUNet(nn.Module):
    """Attention U-Net baseline.

    This baseline uses the same encoder/decoder channel budget as the plain U-Net
    in the benchmark, but inserts attention gates on all four decoder skip paths.
    It is intentionally implemented without pretrained backbones, extra losses,
    or test-time tricks so that it is a fair architecture-only control.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        norm: str = "bn",
        act: str = "relu",
        **_: object,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("AttentionUNet expects five channel values, e.g. [32, 64, 128, 256, 512].")
        c0, c1, c2, c3, c4 = channels

        self.encoder = PyramidEncoder(in_channels=in_channels, channels=channels, block="double", norm=norm, act=act)
        self.dec3 = AttentionUNetDecoderBlock(c4, c3, c3, norm=norm, act=act)
        self.dec2 = AttentionUNetDecoderBlock(c3, c2, c2, norm=norm, act=act)
        self.dec1 = AttentionUNetDecoderBlock(c2, c1, c1, norm=norm, act=act)
        self.dec0 = AttentionUNetDecoderBlock(c1, c0, c0, norm=norm, act=act)
        self.seg_head = nn.Conv2d(c0, num_classes, kernel_size=1)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1, x2, x3, x4 = self.encoder(x)
        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)
        return self.seg_head(d0)


__all__ = ["AttentionUNet", "AttentionUNetDecoderBlock", "AttentionGate"]
