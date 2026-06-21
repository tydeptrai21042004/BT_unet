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
CONFIG_DIR = ROOT / "configs" / "dapr_ablation"

from scripts.dapr_ablation_spec import DAPR_ABLATION_MODELS
from src.models import build_model


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def build(name: str):
    config = load_config(name)
    model_config = dict(config["model"])
    model_config["channels"] = [4, 8, 16, 32, 64]
    model_config["fourier_init_hw"] = [2, 2]
    model_config.pop("name", None)
    return build_model(name, config=model_config)


def test_all_dapr_ablation_configs_exist() -> None:
    assert {path.stem for path in CONFIG_DIR.glob("*.yaml")} == set(DAPR_ABLATION_MODELS)


def test_all_dapr_configs_share_the_same_protocol() -> None:
    configs = [load_config(name) for name in DAPR_ABLATION_MODELS]
    reference = load_config("dapr_unet")
    for config in configs:
        assert config["data"] == reference["data"]
        assert config["train"] == reference["train"]
        assert config["eval"] == reference["eval"]


@pytest.mark.parametrize("name", DAPR_ABLATION_MODELS)
def test_every_dapr_variant_builds_runs_and_backpropagates(name: str) -> None:
    torch.manual_seed(7)
    model = build(name)
    model.train()
    if hasattr(model, "set_epoch"):
        model.set_epoch(1)
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 1, 32, 32)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()


def test_dapr_unet_is_direct_amplitude_phase_reconstruction_only() -> None:
    model = build("dapr_unet")
    assert model.fourier_bottleneck.residual is False
    assert model.fourier_bottleneck.zero_init_output is False
    assert model.fourier_bottleneck.use_amplitude is True
    assert model.fourier_bottleneck.use_phase is True
    assert model.fourier_bottleneck.use_channel_mixing is True
    assert not hasattr(model, "baf_refiner")


def test_dapr_aliases_resolve_to_the_final_proposal() -> None:
    model = build_model(
        "dapr",
        config={
            "channels": [4, 8, 16, 32, 64],
            "fourier_init_hw": [2, 2],
        },
    )
    assert model.__class__.__name__ == "DAPRUNet"
    assert model.fourier_bottleneck.residual is False


def test_dapr_direct_alias_matches_proposal_behavior() -> None:
    torch.manual_seed(123)
    direct = build_model(
        "dapr_direct",
        config={"channels": [4, 8, 16, 32, 64], "fourier_init_hw": [2, 2]},
    )
    torch.manual_seed(123)
    proposed = build_model(
        "dapr_unet",
        config={"channels": [4, 8, 16, 32, 64], "fourier_init_hw": [2, 2]},
    )
    direct_state = direct.state_dict()
    proposed_state = proposed.state_dict()
    assert direct_state.keys() == proposed_state.keys()
    for key in direct_state:
        torch.testing.assert_close(direct_state[key], proposed_state[key])


def test_dapr_ablation_flags_are_active() -> None:
    residual = build("dapr_residual_control")
    assert residual.fourier_bottleneck.residual is True
    assert residual.fourier_bottleneck.zero_init_output is True

    no_phase = build("dapr_no_global_phase")
    assert no_phase.fourier_bottleneck.residual is False
    assert no_phase.fourier_bottleneck.use_phase is False
    assert no_phase.fourier_bottleneck.use_amplitude is True

    no_amp = build("dapr_no_global_amplitude")
    assert no_amp.fourier_bottleneck.residual is False
    assert no_amp.fourier_bottleneck.use_amplitude is False
    assert no_amp.fourier_bottleneck.use_phase is True

    no_mix = build("dapr_no_global_channel_mix")
    assert no_mix.fourier_bottleneck.residual is False
    assert no_mix.fourier_bottleneck.use_channel_mixing is False


def test_dapr_supports_odd_spatial_sizes() -> None:
    model = build("dapr_unet").eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 35, 37))
    assert logits.shape == (1, 1, 35, 37)
    assert torch.isfinite(logits).all()


def test_dapr_report_script_exports_proposed_name(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    latex = tmp_path / "dapr_table.tex"
    delta = tmp_path / "dapr_deltas.csv"
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
        for index, model in enumerate(DAPR_ABLATION_MODELS):
            dice = 0.40 + index * 0.01
            row = {"model": model, "dataset": "etis", "split": "test", "num_seeds": 3, "seeds": "1,2,42"}
            for metric, mean in {
                "dice": dice,
                "iou": dice - 0.08,
                "precision": dice + 0.04,
                "recall": dice + 0.02,
                "mae": 0.08 - index * 0.002,
                "loss": 0.50 - index * 0.01,
            }.items():
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = 0.01
                row[f"{metric}_mean_pm_std"] = f"{mean:.4f} ± 0.0100"
            writer.writerow(row)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_dapr_ablation.py"),
            "--summary-path", str(summary),
            "--latex-path", str(latex),
            "--delta-path", str(delta),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    text = latex.read_text(encoding="utf-8")
    assert "DAPR U-Net (proposed)" in text
    assert "DAPR-BAF" not in text
    assert delta.is_file()
