#!/usr/bin/env python3
"""Build and inspect every retained baseline under fair and faithful contracts."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from baseline_proposal_spec import BASELINE_MODELS, DISPLAY_NAMES  # noqa: E402
from src.engine.output_utils import parse_model_output  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.baselines.attention_unet import (  # noqa: E402
    AttentionGate,
    AttentionUNetDecoderBlock,
)
from src.models.baselines.resunetpp import (  # noqa: E402
    ResUNetPPAttentionGate,
    ResUNetPPDecoderBlock,
)
from src.models.common.blocks import ASPP, ResidualBlock, SqueezeExcitation  # noqa: E402
from src.models.common.decoder import UNetDecoder  # noqa: E402
from src.models.common.encoder import PyramidEncoder  # noqa: E402
from src.models.common.official_backbones import (  # noqa: E402
    OfficialHarDNetEncoder,
    OfficialPVTv2Backbone,
    OfficialRes2NetEncoder,
    OfficialResNet34Encoder,
)
from src.models.common.paper_baselines import (  # noqa: E402
    AdaptiveSelectionModule,
    AxialReverseAttention,
    BoundaryAggregationModule,
    BoundaryPredictionNetwork,
    CFPModule,
    CamouflageIdentificationModule,
    CascadedFusionModule,
    CrossFeatureFusion,
    CrossSemanticAttention,
    DenseAggregation,
    GlobalContextModule,
    HybridSemanticComplementaryModule,
    LocalContextAttention,
    MultiScalePredictionModule,
    RFBModified,
    ReverseAttentionBranch,
    SimilarityAggregationModule,
)

EXPECTED_MODULES: dict[str, tuple[type, ...]] = {
    "unet": (PyramidEncoder, UNetDecoder),
    "unetpp": (PyramidEncoder,),
    "attention_unet": (AttentionGate, AttentionUNetDecoderBlock),
    "pranet": (OfficialRes2NetEncoder, RFBModified, DenseAggregation, ReverseAttentionBranch),
    "acsnet": (OfficialResNet34Encoder, LocalContextAttention, GlobalContextModule, AdaptiveSelectionModule),
    "hardnet_mseg": (OfficialHarDNetEncoder, RFBModified, DenseAggregation),
    "cfanet": (BoundaryPredictionNetwork, CrossFeatureFusion, BoundaryAggregationModule),
    "polyp_pvt": (OfficialPVTv2Backbone, CascadedFusionModule, CamouflageIdentificationModule, SimilarityAggregationModule),
    "caranet": (OfficialRes2NetEncoder, CFPModule, AxialReverseAttention, DenseAggregation),
    "hsnet": (OfficialRes2NetEncoder, OfficialPVTv2Backbone, CrossSemanticAttention, HybridSemanticComplementaryModule, MultiScalePredictionModule),
    "resunetpp": (ResidualBlock, SqueezeExcitation, ASPP, ResUNetPPAttentionGate, ResUNetPPDecoderBlock),
}

IMPLEMENTATION_LEVEL = {
    "unet": "paper-architecture reimplementation",
    "unetpp": "paper-architecture reimplementation",
    "attention_unet": "paper-architecture reimplementation",
    "pranet": "paper-aligned reimplementation with official-compatible Res2Net backbone",
    "acsnet": "paper-aligned reimplementation with official-compatible ResNet-34 backbone",
    "hardnet_mseg": "paper-aligned reimplementation with official-compatible HarDNet-68 backbone",
    "cfanet": "paper-aligned reimplementation with boundary and cross-feature modules",
    "polyp_pvt": "paper-aligned reimplementation with official-compatible PVT-v2 backbone",
    "caranet": "paper-aligned reimplementation with official-compatible Res2Net backbone",
    "hsnet": "paper-aligned dual-backbone reimplementation",
    "resunetpp": "paper-architecture reimplementation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fair-config-dir", default="configs/fair")
    parser.add_argument("--faithful-config-dir", default="configs/official_faithful")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="docs/BASELINE_IMPLEMENTATION_AUDIT.md")
    parser.add_argument("--skip-runtime", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return payload


def runtime_config(name: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the exact fair model configuration used by the benchmark."""
    del name
    return dict(model_cfg)


def main() -> None:
    args = parse_args()
    fair_dir = PROJECT_ROOT / args.fair_config_dir
    faithful_dir = PROJECT_ROOT / args.faithful_config_dir
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)

    expected = {f"{name}.yaml" for name in BASELINE_MODELS}
    fair_files = {path.name for path in fair_dir.glob("*.yaml") if path.stem in BASELINE_MODELS}
    faithful_files = {path.name for path in faithful_dir.glob("*.yaml")}
    errors: list[str] = []
    if fair_files != expected:
        errors.append(f"Fair baseline config set mismatch: {sorted(fair_files)}")
    if faithful_files != expected:
        errors.append(f"Faithful config set mismatch: {sorted(faithful_files)}")

    rows: list[dict[str, object]] = []
    for name in BASELINE_MODELS:
        print(f"[AUDIT] {name}", flush=True)
        fair_cfg = load(fair_dir / f"{name}.yaml")
        faithful_cfg = load(faithful_dir / f"{name}.yaml")
        if fair_cfg.get("model", {}).get("name") != name:
            errors.append(f"{name}: wrong fair model.name")
        if faithful_cfg.get("model", {}).get("name") != name:
            errors.append(f"{name}: wrong faithful model.name")

        row: dict[str, object] = {
            "model": name,
            "display_name": DISPLAY_NAMES[name],
            "implementation_level": IMPLEMENTATION_LEVEL[name],
            "fair_config": "PASS",
            "faithful_config": "PASS",
            "defining_modules": "NOT RUN" if args.skip_runtime else "PASS",
            "forward": "NOT RUN" if args.skip_runtime else "PASS",
            "parameters": "",
        }

        if not args.skip_runtime:
            model_cfg = runtime_config(name, dict(fair_cfg["model"]))
            try:
                model = build_model(name, model_cfg).to(device).eval()
                modules = list(model.modules())
                missing_types = [
                    module_type.__name__
                    for module_type in EXPECTED_MODULES[name]
                    if not any(isinstance(module, module_type) for module in modules)
                ]
                if missing_types:
                    row["defining_modules"] = "FAIL: " + ", ".join(missing_types)
                    errors.append(f"{name}: missing defining modules {missing_types}")

                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                row["parameters"] = parameter_count
                x = torch.randn(1, 3, args.image_size, args.image_size, device=device)
                with torch.no_grad():
                    parsed = parse_model_output(model(x))
                expected_shape = (1, 1, args.image_size, args.image_size)
                if tuple(parsed.main.shape) != expected_shape:
                    row["forward"] = f"FAIL: {tuple(parsed.main.shape)}"
                    errors.append(f"{name}: output shape {tuple(parsed.main.shape)}, expected {expected_shape}")
                elif not torch.isfinite(parsed.main).all():
                    row["forward"] = "FAIL: non-finite output"
                    errors.append(f"{name}: non-finite output")
            except Exception as error:  # noqa: BLE001 - audit must report all models
                row["forward"] = f"FAIL: {type(error).__name__}: {error}"
                errors.append(f"{name}: runtime failure: {type(error).__name__}: {error}")
            finally:
                if "model" in locals():
                    del model
                gc.collect(generation=0)

        rows.append(row)

    lines = [
        "# Baseline implementation audit",
        "",
        "This audit checks the eleven retained manuscript baselines. The repository",
        "contains paper-aligned reimplementations; it does not claim byte-for-byte",
        "identity with every authors' original training repository. Fair-comparison",
        "configs intentionally use one shared loss and optimization protocol, whereas",
        "`configs/official_faithful` preserves paper-style auxiliary/boundary outputs.",
        "",
        "| Baseline | Implementation level | Fair config | Faithful config | Defining modules | Forward | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        parameters = f"{int(row['parameters']):,}" if row["parameters"] != "" else "—"
        lines.append(
            f"| {row['display_name']} | {row['implementation_level']} | "
            f"{row['fair_config']} | {row['faithful_config']} | "
            f"{row['defining_modules']} | {row['forward']} | {parameters} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "PASS" if not errors else "FAIL",
        ]
    )
    if errors:
        lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Baseline audit report: {output}")
    print(f"Baselines checked: {len(rows)}")
    print(f"Errors: {len(errors)}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
