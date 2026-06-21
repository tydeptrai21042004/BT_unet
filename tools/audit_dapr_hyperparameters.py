#!/usr/bin/env python3
"""Audit DAPR hyperparameter-sensitivity configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dapr_hyperparameter_spec import DAPR_HYPERPARAMETER_MODELS

SHARED_SECTIONS = ("data", "train", "eval")
ALLOWED_MODEL_DIFFERENCES = {
    "name",
    "fourier_expansion",
    "fourier_amplitude_scale",
    "fourier_phase_max",
    "fourier_dropout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/dapr_hyperparameters")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return payload


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = ROOT / config_dir

    expected = {f"{name}.yaml" for name in DAPR_HYPERPARAMETER_MODELS}
    actual = {path.name for path in config_dir.glob("*.yaml")}
    file_mismatch = {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}

    configs = {
        name: load(config_dir / f"{name}.yaml")
        for name in DAPR_HYPERPARAMETER_MODELS
        if (config_dir / f"{name}.yaml").is_file()
    }
    metadata_errors = []
    for name, cfg in configs.items():
        if cfg.get("experiment", {}).get("name") != name:
            metadata_errors.append(f"{name}: experiment.name mismatch")
        if cfg.get("model", {}).get("name") != name:
            metadata_errors.append(f"{name}: model.name mismatch")

    protocol_mismatches: dict[str, dict[str, Any]] = {}
    if "dapr_unet" in configs:
        ref = configs["dapr_unet"]
        for section in SHARED_SECTIONS:
            expected_section = ref.get(section, {})
            values = {name: cfg.get(section, {}) for name, cfg in configs.items()}
            if any(value != expected_section for value in values.values()):
                protocol_mismatches[section] = values

        model_mismatch_errors = []
        ref_model = ref.get("model", {})
        for name, cfg in configs.items():
            model = cfg.get("model", {})
            differing = {
                key for key in set(ref_model) | set(model)
                if ref_model.get(key) != model.get(key)
            }
            illegal = sorted(differing - ALLOWED_MODEL_DIFFERENCES)
            if illegal:
                model_mismatch_errors.append(f"{name}: illegal model differences {illegal}")
    else:
        model_mismatch_errors = ["missing default dapr_unet config"]

    ok = not any((
        file_mismatch["missing"],
        file_mismatch["unexpected"],
        metadata_errors,
        protocol_mismatches,
        model_mismatch_errors,
    ))
    report = {
        "config_dir": str(config_dir),
        "models": DAPR_HYPERPARAMETER_MODELS,
        "file_mismatch": file_mismatch,
        "metadata_errors": metadata_errors,
        "protocol_mismatches": protocol_mismatches,
        "model_mismatch_errors": model_mismatch_errors,
        "ok": ok,
    }
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
