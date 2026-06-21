"""Canonical scope and labels for DAPR U-Net hyperparameter sensitivity."""

from __future__ import annotations

DAPR_HYPERPARAMETER_MODELS = [
    "dapr_unet",
    "dapr_expansion_1_0",
    "dapr_expansion_2_0",
    "dapr_amp_scale_0_5",
    "dapr_amp_scale_1_5",
    "dapr_phase_pi_half",
    "dapr_phase_2pi",
    "dapr_dropout_0_0",
]

DISPLAY_NAMES = {
    "dapr_unet": "DAPR U-Net (default)",
    "dapr_expansion_1_0": "Expansion ratio 1.0",
    "dapr_expansion_2_0": "Expansion ratio 2.0",
    "dapr_amp_scale_0_5": "Amplitude scale 0.5",
    "dapr_amp_scale_1_5": "Amplitude scale 1.5",
    "dapr_phase_pi_half": "Phase bound $\\pi/2$",
    "dapr_phase_2pi": "Phase bound $2\\pi$",
    "dapr_dropout_0_0": "Fourier dropout 0.0",
}

DEFAULT_MODEL = "dapr_unet"
SENSITIVITY_COMPARISONS = [
    (model, DEFAULT_MODEL)
    for model in DAPR_HYPERPARAMETER_MODELS
    if model != DEFAULT_MODEL
]

DEFAULT_SEEDS = [42, 1, 2]
DEFAULT_MODELS_CSV = ",".join(DAPR_HYPERPARAMETER_MODELS)
