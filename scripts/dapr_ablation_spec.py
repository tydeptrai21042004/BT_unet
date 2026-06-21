"""Canonical scope and labels for the DAPR U-Net component ablation."""

from __future__ import annotations

DAPR_ABLATION_MODELS = [
    "dapr_residual_control",
    "dapr_no_global_phase",
    "dapr_no_global_amplitude",
    "dapr_no_global_channel_mix",
    "dapr_unet",
]

DISPLAY_NAMES = {
    "dapr_residual_control": "Residual Fourier control",
    "dapr_no_global_phase": "DAPR without phase modulation",
    "dapr_no_global_amplitude": "DAPR without amplitude modulation",
    "dapr_no_global_channel_mix": "DAPR without channel mixing",
    "dapr_unet": "DAPR U-Net (proposed)",
}

COMPONENT_COMPARISONS = [
    (model, "dapr_unet")
    for model in DAPR_ABLATION_MODELS
    if model != "dapr_unet"
]

DEFAULT_SEEDS = [42, 1, 2]
DEFAULT_MODELS_CSV = ",".join(DAPR_ABLATION_MODELS)
