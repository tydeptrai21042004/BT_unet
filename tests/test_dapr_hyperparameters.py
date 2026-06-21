from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "configs" / "dapr_hyperparameters"

from scripts.dapr_hyperparameter_spec import DAPR_HYPERPARAMETER_MODELS  # noqa: E402
from src.models import build_model  # noqa: E402


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def build(name: str):
    config = load_config(name)
    model_config = dict(config["model"])
    model_config["channels"] = [4, 8, 16, 32, 64]
    model_config["fourier_init_hw"] = [2, 2]
    model_config.pop("name", None)
    return build_model(name, config=model_config)


def test_all_hyperparameter_configs_exist() -> None:
    assert {path.stem for path in CONFIG_DIR.glob("*.yaml")} == set(DAPR_HYPERPARAMETER_MODELS)


def test_hyperparameter_configs_share_protocol() -> None:
    configs = [load_config(name) for name in DAPR_HYPERPARAMETER_MODELS]
    reference = load_config("dapr_unet")
    for config in configs:
        assert config["data"] == reference["data"]
        assert config["train"] == reference["train"]
        assert config["eval"] == reference["eval"]


@pytest.mark.parametrize("name", DAPR_HYPERPARAMETER_MODELS)
def test_every_hyperparameter_variant_builds_runs_and_backpropagates(name: str) -> None:
    torch.manual_seed(11)
    model = build(name)
    model.train()
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (1, 1, 32, 32)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_hyperparameter_flags_are_active() -> None:
    assert build("dapr_expansion_1_0").fourier_bottleneck.hidden_channels == 64
    assert build("dapr_expansion_2_0").fourier_bottleneck.hidden_channels == 128
    assert build("dapr_amp_scale_0_5").fourier_bottleneck.amplitude_scale == pytest.approx(0.5)
    assert build("dapr_amp_scale_1_5").fourier_bottleneck.amplitude_scale == pytest.approx(1.5)
    assert build("dapr_phase_pi_half").fourier_bottleneck.phase_max == pytest.approx(torch.pi / 2)
    assert build("dapr_phase_2pi").fourier_bottleneck.phase_max == pytest.approx(2 * torch.pi)
    assert build("dapr_dropout_0_0").fourier_bottleneck.dropout.__class__.__name__ == "Identity"


def test_hyperparameter_report_script_exports_table(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    latex = tmp_path / "hyper.tex"
    delta = tmp_path / "hyper_delta.csv"
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
        for index, model in enumerate(DAPR_HYPERPARAMETER_MODELS):
            dice = 0.50 + index * 0.005
            row = {"model": model, "dataset": "etis", "split": "test", "num_seeds": 3, "seeds": "1,2,42"}
            for metric, mean in {
                "dice": dice,
                "iou": dice - 0.09,
                "precision": dice + 0.02,
                "recall": dice + 0.01,
                "mae": 0.06 - index * 0.001,
                "loss": 0.40 - index * 0.005,
            }.items():
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = 0.01
                row[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± 0.0100"
            writer.writerow(row)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_dapr_hyperparameters.py"),
            "--summary-path", str(summary),
            "--latex-path", str(latex),
            "--delta-path", str(delta),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Hyperparameter sensitivity of DAPR U-Net" in latex.read_text(encoding="utf-8")
    assert delta.is_file()
