from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from baseline_proposal_spec import (  # noqa: E402
    BASELINE_MODELS,
    BASELINE_PROPOSAL_MODELS,
    PROPOSAL_MODELS,
)


def test_exact_baseline_and_dapr_proposal_scope() -> None:
    assert BASELINE_MODELS == [
        "unet",
        "unetpp",
        "attention_unet",
        "pranet",
        "acsnet",
        "hardnet_mseg",
        "cfanet",
        "polyp_pvt",
        "caranet",
        "hsnet",
        "resunetpp",
    ]
    assert PROPOSAL_MODELS == ["dapr_unet"]
    assert len(BASELINE_PROPOSAL_MODELS) == 12


def test_fair_config_directory_matches_dapr_comparison_scope_and_protocol() -> None:
    config_dir = PROJECT_ROOT / "configs" / "fair"
    assert {path.stem for path in config_dir.glob("*.yaml")} == set(BASELINE_PROPOSAL_MODELS)

    configs = {
        name: yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        for name in BASELINE_PROPOSAL_MODELS
    }
    reference = configs["dapr_unet"]
    for name, cfg in configs.items():
        assert cfg["experiment"]["name"] == name
        assert cfg["model"]["name"] == name
        assert cfg["data"] == reference["data"]
        assert cfg["train"] == reference["train"]
        assert cfg["eval"] == reference["eval"]


def test_old_proposals_are_not_in_fair_config_scope() -> None:
    config_dir = PROJECT_ROOT / "configs" / "fair"
    forbidden = {"plain_fourier_unet", "apdr_fourier_unet", "dapr_baf_unet"}
    assert not forbidden & {path.stem for path in config_dir.glob("*.yaml")}


def test_reporting_script_supports_dapr_comparison(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    latex = tmp_path / "comparison.tex"
    delta = tmp_path / "deltas.csv"
    fields = [
        "model", "dataset", "split", "num_seeds", "seeds",
        "dice_mean", "dice_std", "dice_mean_pm_std",
        "iou_mean", "iou_std", "iou_mean_pm_std",
        "precision_mean", "precision_std", "precision_mean_pm_std",
        "recall_mean", "recall_std", "recall_mean_pm_std",
        "mae_mean", "mae_std", "mae_mean_pm_std",
        "loss_mean", "loss_std", "loss_mean_pm_std",
    ]
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, model in enumerate(BASELINE_PROPOSAL_MODELS):
            dice = 0.40 + index * 0.01
            values = {
                "model": model,
                "dataset": "etis",
                "split": "test",
                "num_seeds": 3,
                "seeds": "1,2,42",
            }
            for metric, mean in {
                "dice": dice,
                "iou": dice - 0.08,
                "precision": dice + 0.04,
                "recall": dice + 0.02,
                "mae": 0.08 - index * 0.002,
                "loss": 0.50 - index * 0.01,
            }.items():
                values[f"{metric}_mean"] = mean
                values[f"{metric}_std"] = 0.01
                values[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± 0.0100"
            writer.writerow(values)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "report_baseline_proposal_comparison.py"),
            "--summary-path", str(summary),
            "--latex-path", str(latex),
            "--delta-path", str(delta),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert latex.is_file()
    assert delta.is_file()
    latex_text = latex.read_text(encoding="utf-8")
    assert "DAPR U-Net" in latex_text
    assert "APDR-Fourier U-Net" not in latex_text
    assert "Plain Fourier U-Net" not in latex_text
    assert "DAPR-BAF U-Net" not in latex_text
    with delta.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(PROPOSAL_MODELS) * 6


def test_kaggle_scripts_use_permanent_source_scripts_not_inline_python() -> None:
    for filename in (
        "kaggle_dapr_ablation_etis_3seeds.sh",
        "kaggle_dapr_hyperparameters_etis_3seeds.sh",
        "kaggle_baseline_proposal_etis_3seeds.sh",
    ):
        text = (PROJECT_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "<<'PY'" not in text
        assert '<<"PY"' not in text
        assert "cat >" not in text
        assert "PYTHONDONTWRITEBYTECODE=1" in text
