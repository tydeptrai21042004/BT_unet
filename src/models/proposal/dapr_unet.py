from __future__ import annotations

import math

from ..registry import register_model
from .fourier_unet import _PlainFourierUNetBase


@register_model("dapr_unet")
class DAPRUNet(_PlainFourierUNetBase):
    """DAPR U-Net: Direct Amplitude--Phase Reconstruction U-Net.

    DAPR keeps the U-Net encoder/decoder unchanged, but replaces the deepest
    encoder feature by a directly reconstructed complex Fourier representation.
    Unlike residual Fourier blocks, it does not add the original bottleneck
    feature back to the spectral output.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_residual"] = False
        kwargs["fourier_zero_init_output"] = False
        kwargs.setdefault("fourier_use_amplitude", True)
        kwargs.setdefault("fourier_use_phase", True)
        kwargs.setdefault("fourier_use_channel_mixing", True)
        super().__init__(*args, **kwargs)


@register_model("dapr_residual_control")
class DAPRResidualControl(_PlainFourierUNetBase):
    """Residual Fourier control used to isolate DAPR's direct-reconstruction effect."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_residual"] = True
        kwargs["fourier_zero_init_output"] = True
        kwargs.setdefault("fourier_use_amplitude", True)
        kwargs.setdefault("fourier_use_phase", True)
        kwargs.setdefault("fourier_use_channel_mixing", True)
        super().__init__(*args, **kwargs)


@register_model("dapr_no_global_phase")
class DAPRNoGlobalPhase(DAPRUNet):
    """DAPR ablation with amplitude modulation only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_phase"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_no_global_amplitude")
class DAPRNoGlobalAmplitude(DAPRUNet):
    """DAPR ablation with phase modulation only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_amplitude"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_no_global_channel_mix")
class DAPRNoGlobalChannelMix(DAPRUNet):
    """DAPR ablation without spectral channel mixing."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_channel_mixing"] = False
        super().__init__(*args, **kwargs)


@register_model("dapr_expansion_1_0")
class DAPRExpansion10(DAPRUNet):
    """DAPR sensitivity variant with expansion ratio 1.0."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_expansion"] = 1.0
        super().__init__(*args, **kwargs)


@register_model("dapr_expansion_2_0")
class DAPRExpansion20(DAPRUNet):
    """DAPR sensitivity variant with expansion ratio 2.0."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_expansion"] = 2.0
        super().__init__(*args, **kwargs)


@register_model("dapr_amp_scale_0_5")
class DAPRAmpScale05(DAPRUNet):
    """DAPR sensitivity variant with amplitude scale 0.5."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_amplitude_scale"] = 0.5
        super().__init__(*args, **kwargs)


@register_model("dapr_amp_scale_1_5")
class DAPRAmpScale15(DAPRUNet):
    """DAPR sensitivity variant with amplitude scale 1.5."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_amplitude_scale"] = 1.5
        super().__init__(*args, **kwargs)


@register_model("dapr_phase_pi_half")
class DAPRPhaseHalfPi(DAPRUNet):
    """DAPR sensitivity variant with phase bound pi/2."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_phase_max"] = math.pi / 2.0
        super().__init__(*args, **kwargs)


@register_model("dapr_phase_2pi")
class DAPRPhaseTwoPi(DAPRUNet):
    """DAPR sensitivity variant with phase bound 2pi."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_phase_max"] = 2.0 * math.pi
        super().__init__(*args, **kwargs)


@register_model("dapr_dropout_0_0")
class DAPRDropout00(DAPRUNet):
    """DAPR sensitivity variant without Fourier dropout."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_dropout"] = 0.0
        super().__init__(*args, **kwargs)


__all__ = [
    "DAPRUNet",
    "DAPRResidualControl",
    "DAPRNoGlobalPhase",
    "DAPRNoGlobalAmplitude",
    "DAPRNoGlobalChannelMix",
    "DAPRExpansion10",
    "DAPRExpansion20",
    "DAPRAmpScale05",
    "DAPRAmpScale15",
    "DAPRPhaseHalfPi",
    "DAPRPhaseTwoPi",
    "DAPRDropout00",
]
