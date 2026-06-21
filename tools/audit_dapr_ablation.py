#!/usr/bin/env python3
"""Audit DAPR U-Net one-factor ablation configs and registrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models  # noqa: F401  # populate registry
from scripts.dapr_ablation_spec import DAPR_ABLATION_MODELS
from src.models.registry import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/dapr_ablation")
    return parser.parse_args()


def load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: invalid YAML mapping: {path}")
    return payload


def changed_keys(reference: dict, current: dict) -> set[str]:
    return {
        key
        for key in set(reference) | set(current)
        if reference.get(key) != current.get(key)
    }


def main() -> None:
    args = parse_args()
    config_dir = PROJECT_ROOT / args.config_dir
    expected = set(DAPR_ABLATION_MODELS)
    found = {path.stem for path in config_dir.glob("*.yaml")}
    if found != expected:
        raise SystemExit(
            "ERROR: DAPR configuration scope mismatch.\n"
            f"Expected: {sorted(expected)}\nFound:    {sorted(found)}"
        )
    missing = expected - set(MODEL_REGISTRY)
    if missing:
        raise SystemExit(f"ERROR: unregistered DAPR models: {sorted(missing)}")

    configs = {name: load(config_dir / f"{name}.yaml") for name in DAPR_ABLATION_MODELS}
    reference = configs["dapr_unet"]
    for name, config in configs.items():
        if config.get("model", {}).get("name") != name:
            raise SystemExit(f"ERROR: {name}.yaml has an incorrect model.name")
        for section in ("data", "train", "eval"):
            if config.get(section) != reference.get(section):
                raise SystemExit(f"ERROR: unfair protocol: {name} differs in {section}")

    allowed = {
        "dapr_residual_control": {"name", "fourier_residual", "fourier_zero_init_output"},
        "dapr_no_global_phase": {"name", "fourier_use_phase"},
        "dapr_no_global_amplitude": {"name", "fourier_use_amplitude"},
        "dapr_no_global_channel_mix": {"name", "fourier_use_channel_mixing"},
        "dapr_unet": set(),
    }
    reference_model = reference["model"]
    for name, expected_changes in allowed.items():
        changes = changed_keys(reference_model, configs[name]["model"])
        if changes != expected_changes:
            raise SystemExit(
                f"ERROR: {name} changes {sorted(changes)}; "
                f"expected {sorted(expected_changes)}"
            )

    print("DAPR ablation audit passed.")


if __name__ == "__main__":
    main()
