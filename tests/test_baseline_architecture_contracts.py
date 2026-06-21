from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.output_utils import parse_model_output
from src.models import build_model
from src.models.baselines.attention_unet import AttentionGate, AttentionUNetDecoderBlock
from src.models.baselines.resunetpp import ResUNetPPAttentionGate, ResUNetPPDecoderBlock
from src.models.common.blocks import ASPP, ResidualBlock, SqueezeExcitation
from src.models.common.official_backbones import (
    OfficialHarDNetEncoder, OfficialPVTv2Backbone,
    OfficialRes2NetEncoder, OfficialResNet34Encoder,
)
from src.models.common.paper_baselines import (
    AdaptiveSelectionModule, AxialReverseAttention, BoundaryAggregationModule,
    BoundaryPredictionNetwork, CFPModule, CamouflageIdentificationModule,
    CascadedFusionModule, CrossFeatureFusion, CrossSemanticAttention,
    DenseAggregation, GlobalContextModule, HybridSemanticComplementaryModule,
    LocalContextAttention, MultiScalePredictionModule, RFBModified,
    SimilarityAggregationModule,
)
from src.models.proposal.fourier_unet import FourierSpectralBottleneck
from src.models.proposal.dapr_unet import DAPRUNet

CONFIG_DIR = ROOT / "configs" / "fair"
MODELS = [
    "unet", "unetpp", "attention_unet", "pranet", "acsnet",
    "hardnet_mseg", "cfanet", "polyp_pvt", "caranet", "hsnet",
    "resunetpp", "dapr_unet",
]


def _cfg(name: str) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    model_cfg = dict(cfg["model"])
    if name in {"attention_unet", "dapr_unet"}:
        model_cfg["channels"] = [2, 4, 8, 16, 32]
    if name == "dapr_unet":
        model_cfg["fourier_init_hw"] = [2, 2]
    return model_cfg


CONTRACTS = {
    "attention_unet": (AttentionGate, AttentionUNetDecoderBlock),
    "pranet": (RFBModified, DenseAggregation, OfficialRes2NetEncoder),
    "acsnet": (LocalContextAttention, GlobalContextModule, AdaptiveSelectionModule, OfficialResNet34Encoder),
    "cfanet": (BoundaryPredictionNetwork, CrossFeatureFusion, BoundaryAggregationModule),
    "caranet": (CFPModule, AxialReverseAttention, DenseAggregation, OfficialRes2NetEncoder),
    "hardnet_mseg": (RFBModified, DenseAggregation, OfficialHarDNetEncoder),
    "polyp_pvt": (CascadedFusionModule, CamouflageIdentificationModule, SimilarityAggregationModule, OfficialPVTv2Backbone),
    "hsnet": (CrossSemanticAttention, HybridSemanticComplementaryModule, MultiScalePredictionModule, OfficialRes2NetEncoder, OfficialPVTv2Backbone),
    "resunetpp": (ResidualBlock, SqueezeExcitation, ASPP, ResUNetPPAttentionGate, ResUNetPPDecoderBlock),
    "dapr_unet": (FourierSpectralBottleneck, DAPRUNet),
}


@pytest.mark.parametrize("name", MODELS)
def test_fair_config_builds_and_preserves_shape(name: str) -> None:
    model = build_model(name, _cfg(name)).eval()
    with torch.no_grad():
        parsed = parse_model_output(model(torch.randn(1, 3, 32, 32)))
    assert parsed.main.shape == (1, 1, 32, 32)
    assert torch.isfinite(parsed.main).all()


@pytest.mark.parametrize("name,module_types", CONTRACTS.items())
def test_models_expose_defining_architecture_modules(name: str, module_types: tuple[type, ...]) -> None:
    modules = list(build_model(name, _cfg(name)).modules())
    for module_type in module_types:
        assert any(isinstance(module, module_type) for module in modules), (name, module_type.__name__)


def test_fairness_audit_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "audit_fairness.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"fair": true' in proc.stdout.lower()
