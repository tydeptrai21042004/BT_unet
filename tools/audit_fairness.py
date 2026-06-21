from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from baseline_proposal_spec import BASELINE_PROPOSAL_MODELS

CONFIG_DIR = ROOT / "configs" / "fair"
MODELS = BASELINE_PROPOSAL_MODELS
SHARED_PATHS = [
    ("data", "augmentation"), ("data", "batch_size"),
    ("data", "image_size"), ("data", "num_workers"),
    ("data", "pin_memory"), ("train", "epochs"),
    ("train", "lr"), ("train", "weight_decay"),
    ("train", "optimizer"), ("train", "scheduler"),
    ("train", "t_max"), ("train", "eta_min"),
    ("train", "mixed_precision"), ("train", "deterministic"),
    ("train", "grad_clip"), ("train", "loss"),
    ("train", "threshold"), ("train", "use_aux_outputs_loss"),
    ("train", "use_boundary_loss"),
    ("train", "gradient_accumulation_steps"),
    ("eval", "loss"), ("eval", "threshold"),
]


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    configs = {name: load(CONFIG_DIR / f"{name}.yaml") for name in MODELS}
    reference = configs["dapr_unet"]
    strict_mismatches: dict[str, dict[str, Any]] = {}
    for section, key in SHARED_PATHS:
        values = {name: cfg.get(section, {}).get(key) for name, cfg in configs.items()}
        if any(value != reference[section][key] for value in values.values()):
            strict_mismatches[f"{section}.{key}"] = values

    report = {
        "models": MODELS,
        "strict_mismatches": strict_mismatches,
        "fair": not strict_mismatches,
        "notes": [
            "All baselines and DAPR use the same split, augmentation, optimizer, loss, epochs, threshold, and seeds.",
            "Architecture-specific auxiliary losses are disabled for every method in the fair comparison.",
            "DAPR U-Net is the only active proposal method.",
        ],
    }
    print(json.dumps(report, indent=2))
    if not report["fair"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
