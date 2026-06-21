"""Canonical model lists used by all benchmark entrypoints."""

BASELINE_MODELS = ['unet', 'unetpp', 'attention_unet', 'pranet', 'acsnet', 'hardnet_mseg', 'cfanet', 'polyp_pvt', 'caranet', 'hsnet', 'resunetpp']

PROPOSAL_MODELS = ["dapr_unet"]

FAIR_MODELS = BASELINE_MODELS + PROPOSAL_MODELS
BASELINE_PROPOSAL_MODELS = FAIR_MODELS
ABLATION_MODELS = ["dapr_unet"]
