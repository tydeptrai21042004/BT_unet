from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.decoder import UNetDecoder
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights


class FourierSpectralBottleneck(nn.Module):
    """Amplitude--phase Fourier mixer used by DAPR and its residual control.

    The implementation preserves the successful baseline behavior: the
    spectral correction is projected back to the original channel count and
    added residually.  ``forward_with_disagreement`` is kept for diagnostic compatibility only.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.5,
        alpha: float = 0.5,
        dropout: float = 0.1,
        norm: str = "gn",
        act: str = "gelu",
        init_hw: Sequence[int] = (22, 22),
        amplitude_scale: float = 1.0,
        phase_max: float = math.pi,
        use_amplitude: bool = True,
        use_phase: bool = True,
        use_channel_mixing: bool = True,
        residual: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if len(init_hw) != 2 or min(int(init_hw[0]), int(init_hw[1])) <= 0:
            raise ValueError("init_hw must contain two positive integers")
        if amplitude_scale < 0 or phase_max < 0:
            raise ValueError("spectral response limits must be non-negative")
        if not (use_amplitude or use_phase or use_channel_mixing):
            raise ValueError("At least one spectral operation must be enabled")
        if not residual and zero_init_output:
            raise ValueError("zero_init_output requires residual=True")

        hidden = max(int(channels), int(round(channels * float(expansion))))
        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.alpha = float(alpha)
        self.amplitude_scale = float(amplitude_scale)
        self.phase_max = float(phase_max)
        self.use_amplitude = bool(use_amplitude)
        self.use_phase = bool(use_phase)
        self.use_channel_mixing = bool(use_channel_mixing)
        self.residual = bool(residual)
        self.zero_init_output = bool(zero_init_output)

        self.pre = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pre_norm = self._make_norm(norm, hidden)
        self.pre_act = self._make_act(act)

        h0, w0 = int(init_hw[0]), int(init_hw[1])
        self.amplitude_logits = nn.Parameter(torch.empty(1, hidden, h0, w0 // 2 + 1))
        self.phase_logits = nn.Parameter(torch.empty(1, hidden, h0, w0 // 2 + 1))

        if self.use_channel_mixing:
            self.spectral_channel_mixer: nn.Module = nn.Conv2d(hidden, hidden, 1, bias=False)
        else:
            self.spectral_channel_mixer = nn.Identity()

        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.post = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.reset_parameters()

    @staticmethod
    def _make_norm(kind: str, channels: int) -> nn.Module:
        key = str(kind).lower()
        if key in {"none", "identity", ""}:
            return nn.Identity()
        if key == "bn":
            return nn.BatchNorm2d(channels)
        if key == "gn":
            groups = min(32, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, channels)
        raise ValueError(f"Unsupported norm: {kind}")

    @staticmethod
    def _make_act(kind: str) -> nn.Module:
        key = str(kind).lower()
        if key in {"none", "identity", ""}:
            return nn.Identity()
        if key == "relu":
            return nn.ReLU(inplace=True)
        if key == "gelu":
            return nn.GELU()
        if key in {"silu", "swish"}:
            return nn.SiLU(inplace=True)
        raise ValueError(f"Unsupported activation: {kind}")

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.pre.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.amplitude_logits, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.phase_logits, mean=0.0, std=1.0e-3)
        if isinstance(self.spectral_channel_mixer, nn.Conv2d):
            nn.init.zeros_(self.spectral_channel_mixer.weight)
            eye = torch.eye(self.hidden_channels, dtype=self.spectral_channel_mixer.weight.dtype)
            self.spectral_channel_mixer.weight.data[:, :, 0, 0].copy_(eye)
        if self.zero_init_output:
            nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.post.bias)
        else:
            nn.init.kaiming_normal_(self.post.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.post.bias)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    @staticmethod
    def _resize_parameter(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def _response(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequency_size = spectrum.shape[-2:]
        if self.use_amplitude:
            logits = self._resize_parameter(self.amplitude_logits, frequency_size)
            gain = torch.exp(self.amplitude_scale * torch.tanh(logits))
        else:
            gain = torch.ones_like(spectrum.real)
        if self.use_phase:
            logits = self._resize_parameter(self.phase_logits, frequency_size)
            phase = self.phase_max * torch.tanh(logits)
        else:
            phase = torch.zeros_like(spectrum.real)
        return gain, phase

    def _mix(self, spectrum: torch.Tensor) -> torch.Tensor:
        if not self.use_channel_mixing:
            return spectrum
        return torch.complex(
            self.spectral_channel_mixer(spectrum.real),
            self.spectral_channel_mixer(spectrum.imag),
        )

    def _forward_internal(
        self,
        x: torch.Tensor,
        *,
        return_disagreement: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.pre_act(self.pre_norm(self.pre(x)))
        spatial_size = z.shape[-2:]
        spectrum = torch.fft.rfft2(z, dim=(-2, -1), norm="ortho")
        gain, phase = self._response(spectrum)
        transformed = self._mix(spectrum * torch.polar(gain, phase))

        if self.residual:
            signal = torch.fft.irfft2(
                transformed - spectrum,
                s=spatial_size,
                dim=(-2, -1),
                norm="ortho",
            )
            output = x + self.alpha * self.post(self.dropout(signal))
        else:
            signal = torch.fft.irfft2(
                transformed,
                s=spatial_size,
                dim=(-2, -1),
                norm="ortho",
            )
            output = self.post(self.dropout(signal))

        disagreement: torch.Tensor | None = None
        if return_disagreement:
            # Channel mixing is intentionally excluded here.  Both evidence
            # paths therefore differ only in amplitude versus phase response.
            amp_only = spectrum * torch.polar(gain, torch.zeros_like(phase))
            phase_only = spectrum * torch.polar(torch.ones_like(gain), phase)
            amp_signal = torch.fft.irfft2(
                amp_only - spectrum,
                s=spatial_size,
                dim=(-2, -1),
                norm="ortho",
            )
            phase_signal = torch.fft.irfft2(
                phase_only - spectrum,
                s=spatial_size,
                dim=(-2, -1),
                norm="ortho",
            )
            disagreement = (amp_signal - phase_signal).abs().mean(dim=1, keepdim=True)
            scale = disagreement.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
            disagreement = torch.tanh(disagreement / scale)

        return output, disagreement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self._forward_internal(x, return_disagreement=False)
        return output

    def forward_with_disagreement(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, disagreement = self._forward_internal(x, return_disagreement=True)
        assert disagreement is not None
        return output, disagreement

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        gain = torch.exp(self.amplitude_scale * torch.tanh(self.amplitude_logits))
        phase = self.phase_max * torch.tanh(self.phase_logits)
        return {
            "fourier/alpha": float(self.alpha),
            "fourier/amplitude_gain_mean": float(gain.mean().item()),
            "fourier/amplitude_gain_min": float(gain.min().item()),
            "fourier/amplitude_gain_max": float(gain.max().item()),
            "fourier/phase_abs_mean": float(phase.abs().mean().item()),
            "fourier/phase_abs_max": float(phase.abs().max().item()),
        }


class _PlainFourierUNetBase(nn.Module):
    """Shared U-Net path with a configurable Fourier bottleneck."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        fourier_alpha: float = 0.5,
        fourier_alpha_start: float = 0.5,
        fourier_alpha_warmup_epochs: int = 0,
        fourier_expansion: float = 1.5,
        fourier_dropout: float = 0.1,
        fourier_block_norm: str = "gn",
        fourier_block_act: str = "gelu",
        fourier_init_hw: Sequence[int] = (22, 22),
        fourier_amplitude_scale: float = 1.0,
        fourier_phase_max: float = math.pi,
        fourier_use_amplitude: bool = True,
        fourier_use_phase: bool = True,
        fourier_use_channel_mixing: bool = True,
        fourier_residual: bool = True,
        fourier_zero_init_output: bool = True,
        norm: str = "bn",
        act: str = "relu",
        decoder_use_cbam: bool = False,
        fourier_stage_index: int = -1,
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in channels)
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            channels=channels,
            block="double",
            norm=norm,
            act=act,
        )
        self.fourier_alpha_target = float(fourier_alpha)
        self.fourier_alpha_start = float(fourier_alpha_start)
        self.fourier_alpha_warmup_epochs = int(fourier_alpha_warmup_epochs)
        self.fourier_stage_index = int(fourier_stage_index)
        if self.fourier_stage_index < 0:
            self.fourier_stage_index += len(channels)
        if not 0 <= self.fourier_stage_index < len(channels):
            raise ValueError("fourier_stage_index selects an invalid encoder feature")

        self.fourier_bottleneck = FourierSpectralBottleneck(
            channels=channels[self.fourier_stage_index],
            expansion=fourier_expansion,
            alpha=fourier_alpha,
            dropout=fourier_dropout,
            norm=fourier_block_norm,
            act=fourier_block_act,
            init_hw=fourier_init_hw,
            amplitude_scale=fourier_amplitude_scale,
            phase_max=fourier_phase_max,
            use_amplitude=fourier_use_amplitude,
            use_phase=fourier_use_phase,
            use_channel_mixing=fourier_use_channel_mixing,
            residual=fourier_residual,
            zero_init_output=fourier_zero_init_output,
        )
        self.decoder = UNetDecoder(channels=channels, norm=norm, act=act, use_cbam=decoder_use_cbam)
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        init_weights(self)
        self.fourier_bottleneck.reset_parameters()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.fourier_alpha_warmup_epochs <= 0:
            alpha = self.fourier_alpha_target
        else:
            progress = min(max(float(epoch), 0.0) / float(self.fourier_alpha_warmup_epochs), 1.0)
            alpha = self.fourier_alpha_start + (
                self.fourier_alpha_target - self.fourier_alpha_start
            ) * progress
        self.fourier_bottleneck.set_alpha(alpha)

    def _plain_forward_features(
        self,
        x: torch.Tensor,
        *,
        return_disagreement: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.encoder(x)
        if return_disagreement:
            transformed, disagreement = self.fourier_bottleneck.forward_with_disagreement(
                features[self.fourier_stage_index]
            )
        else:
            transformed = self.fourier_bottleneck(features[self.fourier_stage_index])
            disagreement = None
        features[self.fourier_stage_index] = transformed
        return self.decoder(features), disagreement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded, _ = self._plain_forward_features(x, return_disagreement=False)
        return self.seg_head(decoded)

    def auxiliary_regularization(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def diagnostic_metrics(self) -> dict[str, float]:
        return self.fourier_bottleneck.diagnostics()


class ResidualFourierUNetControl(_PlainFourierUNetBase):
    """Unregistered residual Fourier control used only inside DAPR ablations.

    This class is intentionally not registered as an active proposal method;
    the repository keeps DAPR U-Net as the only manuscript proposal.
    """


__all__ = [
    "FourierSpectralBottleneck",
    "_PlainFourierUNetBase",
    "ResidualFourierUNetControl",
]
