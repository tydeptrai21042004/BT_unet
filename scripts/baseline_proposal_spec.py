"""Canonical model sets and labels for baseline-versus-DAPR experiments."""

from __future__ import annotations

BASELINE_MODELS = ['unet', 'unetpp', 'attention_unet', 'pranet', 'acsnet', 'hardnet_mseg', 'cfanet', 'polyp_pvt', 'caranet', 'hsnet', 'resunetpp']

PROPOSAL_MODELS = ["dapr_unet"]
BASELINE_PROPOSAL_MODELS = BASELINE_MODELS + PROPOSAL_MODELS
DEFAULT_MODELS_CSV = ",".join(BASELINE_PROPOSAL_MODELS)
DEFAULT_SEEDS = [42, 1, 2]

DISPLAY_NAMES = {
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net",
    "pranet": "PraNet",
    "acsnet": "ACSNet",
    "hardnet_mseg": "HarDNet-MSEG",
    "cfanet": "CFANet",
    "polyp_pvt": "Polyp-PVT",
    "caranet": "CaraNet",
    "hsnet": "HSNet",
    "resunetpp": "ResUNet++",
    "dapr_unet": "DAPR U-Net (proposed)",
}

GROUPS = {
    **{model: "baseline" for model in BASELINE_MODELS},
    **{model: "proposal" for model in PROPOSAL_MODELS},
}
