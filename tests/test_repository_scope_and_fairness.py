from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIGS = ROOT / "configs"

import src.models  # noqa: E402,F401
from src.models.registry import MODEL_REGISTRY  # noqa: E402

BASELINES = {
    "unet", "unetpp", "attention_unet", "pranet", "acsnet",
    "hardnet_mseg", "cfanet", "polyp_pvt", "caranet", "hsnet",
    "resunetpp",
}
PROPOSALS = {"dapr_unet"}
DAPR_ABLATION = {
    "dapr_residual_control", "dapr_no_global_phase",
    "dapr_no_global_amplitude", "dapr_no_global_channel_mix",
    "dapr_unet",
}
DAPR_HYPERPARAMETERS = {
    "dapr_unet", "dapr_expansion_1_0", "dapr_expansion_2_0",
    "dapr_amp_scale_0_5", "dapr_amp_scale_1_5",
    "dapr_phase_pi_half", "dapr_phase_2pi", "dapr_dropout_0_0",
}
REGISTRY_MODELS = BASELINES | DAPR_ABLATION | DAPR_HYPERPARAMETERS
FORBIDDEN_OLD_PROPOSALS = {
    "plain_fourier_unet", "apdr_fourier_unet", "dapr_baf_unet",
    "dapr_direct_unet", "apdr_uniform_route", "plain_fourier_no_residual",
}


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_registry_contains_only_requested_models() -> None:
    assert set(MODEL_REGISTRY) == REGISTRY_MODELS
    assert not FORBIDDEN_OLD_PROPOSALS & set(MODEL_REGISTRY)


def test_config_directories_have_exact_scope() -> None:
    assert {p.name for p in CONFIGS.iterdir() if p.is_dir()} == {
        "official_faithful", "fair", "dapr_ablation", "dapr_hyperparameters"
    }
    assert {p.stem for p in (CONFIGS / "official_faithful").glob("*.yaml")} == BASELINES
    assert {p.stem for p in (CONFIGS / "fair").glob("*.yaml")} == BASELINES | PROPOSALS
    assert {p.stem for p in (CONFIGS / "dapr_ablation").glob("*.yaml")} == DAPR_ABLATION
    assert {p.stem for p in (CONFIGS / "dapr_hyperparameters").glob("*.yaml")} == DAPR_HYPERPARAMETERS


def test_all_fair_configs_share_the_same_training_and_evaluation_protocol() -> None:
    configs = {p.stem: load(p) for p in (CONFIGS / "fair").glob("*.yaml")}
    reference = configs["dapr_unet"]
    shared_paths = [
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
    for section, key in shared_paths:
        expected = reference[section][key]
        for name, cfg in configs.items():
            assert cfg[section][key] == expected, (name, section, key)


def test_dapr_component_ablation_changes_only_dapr_model_keys() -> None:
    configs = {p.stem: load(p) for p in (CONFIGS / "dapr_ablation").glob("*.yaml")}
    reference = configs["dapr_unet"]
    allowed = {"name", "fourier_residual", "fourier_zero_init_output", "fourier_use_phase", "fourier_use_amplitude", "fourier_use_channel_mixing"}
    for name, cfg in configs.items():
        assert cfg["data"] == reference["data"]
        assert cfg["train"] == reference["train"]
        assert cfg["eval"] == reference["eval"]
        differing = {
            key for key in set(reference["model"]) | set(cfg["model"])
            if reference["model"].get(key) != cfg["model"].get(key)
        }
        assert differing <= allowed, (name, differing - allowed)


def test_dapr_hyperparameter_analysis_changes_only_hyperparameter_keys() -> None:
    configs = {p.stem: load(p) for p in (CONFIGS / "dapr_hyperparameters").glob("*.yaml")}
    reference = configs["dapr_unet"]
    allowed = {"name", "fourier_expansion", "fourier_amplitude_scale", "fourier_phase_max", "fourier_dropout"}
    for name, cfg in configs.items():
        assert cfg["data"] == reference["data"]
        assert cfg["train"] == reference["train"]
        assert cfg["eval"] == reference["eval"]
        differing = {
            key for key in set(reference["model"]) | set(cfg["model"])
            if reference["model"].get(key) != cfg["model"].get(key)
        }
        assert differing <= allowed, (name, differing - allowed)


def test_active_code_no_longer_references_old_proposal_methods() -> None:
    forbidden = (
        "apdr_fourier_unet", "dapr_baf_unet", "plain_fourier_unet",
        "apdr_uniform_route", "dapr_direct_unet",
    )
    roots = [ROOT / "src", ROOT / "configs", ROOT / "scripts", ROOT / "README.md"]
    allowed_files = {
        "tests/test_repository_scope_and_fairness.py",
    }
    hits: list[tuple[str, str]] = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".tex", ".yaml", ".yml", ".sh"}:
                continue
            rel = str(path.relative_to(ROOT))
            if rel in allowed_files:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in content:
                    hits.append((rel, token))
    assert hits == []


def test_manuscript_baseline_fragment_contains_exact_requested_rows() -> None:
    text = (ROOT / "docs" / "manuscript_baselines.tex").read_text(encoding="utf-8")
    required = [
        "U-Net", "U-Net++", "Attention U-Net", "PraNet", "ACSNet",
        "HarDNet-MSEG", "CFANet", "Polyp-PVT", "CaraNet", "HSNet",
        "ResUNet++",
    ]
    for name in required:
        assert name in text
    assert text.count("\\\\") == 11
