from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34

from .utils import ensure_tuple_channels
from ..vendor.hardnet import HarDBlock  # re-exported for tests/contracts
from ..vendor.res2net_v1b import Res2Net, Bottle2neck, res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from ..vendor.pvt_v2_compat import pvt_v2_b0, pvt_v2_b0_fast, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_fast
from ..vendor.hardnet import HarDNet


DEFAULT_CHECKPOINT_URLS = {
    "res2net50_v1b_26w_4s": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
    "pvt_v2_b2": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
    "hardnet68": "https://ping-chao.com/hardnet/hardnet68-5d684880.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
}


def _unwrap_checkpoint(state):
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state


def _clean_key(key: str) -> str:
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _load_state_dict(model: nn.Module, checkpoint: str | None = None, url: str | None = None, strict: bool = False) -> bool:
    """Load a checkpoint with shape-safe filtering.

    Public backbone checkpoints often differ slightly from local adapter key names
    or include classifier heads. Shape filtering prevents an ImageNet classifier
    weight from crashing a segmentation benchmark run while still loading all
    compatible official backbone parameters.
    """
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            warnings.warn(f"Checkpoint not found: {checkpoint}")
            return False
        state = torch.load(path, map_location="cpu")
    elif url:
        try:
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
        except Exception as exc:  # pragma: no cover - network-dependent
            warnings.warn(f"Failed to download checkpoint from {url}: {exc}")
            return False
    else:
        return False

    state = _unwrap_checkpoint(state)
    if not isinstance(state, dict):
        warnings.warn(f"Unsupported checkpoint payload type: {type(state)!r}")
        return False

    model_state = model.state_dict()
    cleaned = {_clean_key(str(k)): v for k, v in state.items() if torch.is_tensor(v)}
    compatible = {k: v for k, v in cleaned.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
    skipped = len(cleaned) - len(compatible)

    if not compatible:
        warnings.warn("No compatible tensors found in checkpoint; leaving backbone randomly initialized.")
        return False

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        warnings.warn(f"Skipped {skipped} incompatible checkpoint tensors while loading official backbone.")
    if strict and (missing or unexpected):
        warnings.warn(f"Strict checkpoint load requested but missing={len(missing)} unexpected={len(unexpected)}.")
    return True


class _Projection(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        if in_ch == out_ch:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OfficialResNet34Encoder(nn.Module):
    """ResNet-34 encoder used by the official ACSNet implementation.

    Returns five features at [H/2, H/4, H/8, H/16, H/32]. Projection layers
    preserve the benchmark channel contract without changing the ResNet-34
    stage topology.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("OfficialResNet34Encoder expects five output channel values.")
        if in_channels != 3:
            raise ValueError("OfficialResNet34Encoder currently supports RGB input only.")
        self.backbone = resnet34(weights=None)
        if pretrained and not checkpoint and not checkpoint_url:
            checkpoint_url = DEFAULT_CHECKPOINT_URLS["resnet34"]
        if checkpoint or checkpoint_url:
            loaded = _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
            if pretrained and not loaded:
                warnings.warn("Requested ResNet-34 pretrained=True, but checkpoint loading failed; continuing from random initialization.")
        self.raw_channels = (64, 64, 128, 256, 512)
        self.channels = channels
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(self.raw_channels, channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        return [proj(feat) for proj, feat in zip(self.projections, (x0, x1, x2, x3, x4))]


class OfficialRes2NetEncoder(nn.Module):
    """Official Res2Net backbone adapter with projected multi-scale outputs.

    Returns five features matching the benchmark contract: [H/2, H/4, H/8, H/16, H/32].
    If ``pretrained=True`` and no explicit checkpoint path/URL is supplied, the
    adapter automatically downloads the public Res2Net-50 ImageNet checkpoint.
    """

    VARIANTS = {
        "res2net50_v1b_26w_4s": (lambda pretrained=False: res2net50_v1b_26w_4s(pretrained=pretrained), (64, 256, 512, 1024, 2048)),
        "res2net101_v1b_26w_4s": (lambda pretrained=False: res2net101_v1b_26w_4s(pretrained=pretrained), (64, 256, 512, 1024, 2048)),
        "res2net50_v1b_26w_4s_fast": (lambda pretrained=False: Res2Net(Bottle2neck, [1, 1, 1, 1], baseWidth=26, scale=4), (64, 256, 512, 1024, 2048)),
    }

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        variant: str = "res2net50_v1b_26w_4s",
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("OfficialRes2NetEncoder expects five output channel values.")
        if in_channels != 3:
            raise ValueError("OfficialRes2NetEncoder currently supports RGB input only.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported Res2Net variant: {variant}")
        ctor, raw_channels = self.VARIANTS[variant]
        self.backbone = ctor(pretrained=False)
        if pretrained and not checkpoint and not checkpoint_url:
            checkpoint_url = DEFAULT_CHECKPOINT_URLS.get(variant)
        if checkpoint or checkpoint_url:
            loaded = _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
            if pretrained and not loaded:
                warnings.warn("Requested Res2Net pretrained=True, but checkpoint loading failed; continuing from random initialization.")
        self.channels = channels
        self.raw_channels = raw_channels
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(raw_channels, channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        feats = [x0, x1, x2, x3, x4]
        return [proj(feat) for proj, feat in zip(self.projections, feats)]


class OfficialPVTv2Backbone(nn.Module):
    """Official PVTv2 backbone adapter with projected stage outputs.

    If ``pretrained=True`` and no explicit checkpoint path/URL is supplied, the
    adapter automatically downloads the official PVTv2-B2 ImageNet checkpoint
    when that variant is used.
    """

    VARIANTS = {
        "pvt_v2_b0": (pvt_v2_b0, (32, 64, 160, 256)),
        "pvt_v2_b0_fast": (pvt_v2_b0_fast, (32, 64, 160, 256)),
        "pvt_v2_b1": (pvt_v2_b1, (64, 128, 320, 512)),
        "pvt_v2_b2": (pvt_v2_b2, (64, 128, 320, 512)),
        "pvt_v2_b2_fast": (pvt_v2_b2_fast, (64, 128, 320, 512)),
    }

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: Sequence[int] = (32, 64, 128, 256),
        variant: str = "pvt_v2_b2",
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
        image_size: int = 352,
    ) -> None:
        super().__init__()
        embed_dims = ensure_tuple_channels(embed_dims)
        if len(embed_dims) != 4:
            raise ValueError("OfficialPVTv2Backbone expects four output stage channel values.")
        if in_channels != 3:
            raise ValueError("OfficialPVTv2Backbone currently supports RGB input only.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported PVTv2 variant: {variant}")
        ctor, raw_channels = self.VARIANTS[variant]
        self.backbone = ctor(img_size=image_size, in_chans=in_channels)
        if pretrained and not checkpoint and not checkpoint_url:
            checkpoint_url = DEFAULT_CHECKPOINT_URLS.get(variant)
        if checkpoint or checkpoint_url:
            loaded = _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
            if pretrained and not loaded:
                warnings.warn("Requested PVTv2 pretrained=True, but checkpoint loading failed; continuing from random initialization.")
        self.channels = tuple(int(c) for c in embed_dims)
        self.raw_channels = raw_channels
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(raw_channels, self.channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone.forward_features(x)
        return [proj(feat) for proj, feat in zip(self.projections, feats)]


class OfficialHarDNetEncoder(nn.Module):
    """Official HarDNet-68 backbone adapter with projected outputs."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        arch: int = 68,
        pretrained: bool = False,
        checkpoint: str | None = None,
        checkpoint_url: str | None = None,
    ) -> None:
        super().__init__()
        channels = ensure_tuple_channels(channels)
        if len(channels) != 5:
            raise ValueError("OfficialHarDNetEncoder expects five output channel values.")
        if in_channels != 3:
            raise ValueError("OfficialHarDNetEncoder currently supports RGB input only.")
        self.backbone = HarDNet(arch=arch, pretrained=False)
        if pretrained and not checkpoint and not checkpoint_url:
            checkpoint_url = DEFAULT_CHECKPOINT_URLS.get(f"hardnet{arch}")
        if checkpoint or checkpoint_url:
            loaded = _load_state_dict(self.backbone, checkpoint=checkpoint, url=checkpoint_url, strict=False)
            if pretrained and not loaded:
                warnings.warn("Requested HarDNet pretrained=True, but checkpoint loading failed; continuing from random initialization.")
        self.raw_channels = (32, 128, 256, 640, 1024)
        self.channels = tuple(int(c) for c in channels)
        self.projections = nn.ModuleList(_Projection(ic, oc) for ic, oc in zip(self.raw_channels, self.channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        base = self.backbone.base
        s0 = base[0](x)
        x = base[1](s0)
        x = base[2](x)
        x = base[3](x)
        s1 = base[4](x)
        x = base[5](s1)
        x = base[6](x)
        s2 = base[7](x)
        x = base[8](s2)
        x = base[9](x)
        x = base[10](x)
        x = base[11](x)
        s3 = base[12](x)
        x = base[13](s3)
        x = base[14](x)
        s4 = base[15](x)
        feats = [s0, s1, s2, s3, s4]
        return [proj(feat) for proj, feat in zip(self.projections, feats)]


__all__ = [
    "DEFAULT_CHECKPOINT_URLS",
    "OfficialResNet34Encoder",
    "OfficialRes2NetEncoder",
    "OfficialPVTv2Backbone",
    "OfficialHarDNetEncoder",
    "HarDBlock",
]
