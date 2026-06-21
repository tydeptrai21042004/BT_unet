#!/usr/bin/env python3
"""Strict baseline-vs-DAPR audit with registry and forward-pass checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import src.models  # noqa: F401,E402
from baseline_proposal_spec import BASELINE_MODELS, BASELINE_PROPOSAL_MODELS, PROPOSAL_MODELS
from src.models import build_model
from src.models.registry import MODEL_REGISTRY

FORBIDDEN_REGISTERED_PROPOSALS = {
    "plain_fourier_unet",
    "apdr_fourier_unet",
    "dapr_baf_unet",
    "dapr_direct_unet",
    "apdr_uniform_route",
    "plain_fourier_no_residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/fair")
    parser.add_argument("--forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def tiny_model_config(cfg: dict) -> dict:
    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    model_cfg["channels"] = [4, 8, 16, 32, 64]
    if "fourier_init_hw" in model_cfg:
        model_cfg["fourier_init_hw"] = [2, 2]
    return model_cfg


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = ROOT / config_dir

    errors: list[str] = []
    if PROPOSAL_MODELS != ["dapr_unet"]:
        errors.append(f"Expected only dapr_unet as proposal, got {PROPOSAL_MODELS}")
    if len(BASELINE_MODELS) != 11:
        errors.append(f"Expected 11 baselines, got {len(BASELINE_MODELS)}")
    if len(BASELINE_PROPOSAL_MODELS) != 12:
        errors.append(f"Expected 12 fair-comparison models, got {len(BASELINE_PROPOSAL_MODELS)}")

    forbidden_present = sorted(FORBIDDEN_REGISTERED_PROPOSALS & set(MODEL_REGISTRY))
    if forbidden_present:
        errors.append(f"Old proposal names still registered: {forbidden_present}")

    expected_files = {f"{name}.yaml" for name in BASELINE_PROPOSAL_MODELS}
    actual_files = {path.name for path in config_dir.glob("*.yaml")}
    if expected_files != actual_files:
        errors.append(f"Fair config files mismatch: missing={sorted(expected_files-actual_files)}, unexpected={sorted(actual_files-expected_files)}")

    forward_results = {}
    if args.forward:
        x = torch.randn(1, 3, 32, 32)
        for name in BASELINE_PROPOSAL_MODELS:
            cfg_path = config_dir / f"{name}.yaml"
            if not cfg_path.is_file():
                continue
            cfg = load_yaml(cfg_path)
            torch.manual_seed(123)
            model = build_model(name, config=tiny_model_config(cfg)).eval()
            with torch.no_grad():
                y = model(x)
            ok = isinstance(y, torch.Tensor) and y.shape == (1, 1, 32, 32) and torch.isfinite(y).all().item()
            forward_results[name] = {"ok": bool(ok), "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            if not ok:
                errors.append(f"Forward check failed for {name}")

    report = {
        "baselines": BASELINE_MODELS,
        "proposals": PROPOSAL_MODELS,
        "registered_models": sorted(MODEL_REGISTRY),
        "forward_results": forward_results,
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
