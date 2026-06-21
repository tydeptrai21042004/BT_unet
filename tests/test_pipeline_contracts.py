from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import Evaluator, Trainer
from src.losses import BCEDiceLoss
from src.models import build_model


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.rand(3, 32, 32, generator=generator),
            "mask": (torch.rand(1, 32, 32, generator=generator) > 0.5).float(),
        }


def _small_config(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "dapr_ablation" / f"{name}.yaml").read_text())
    model_cfg = dict(cfg["model"])
    model_cfg["channels"] = [4, 8, 16, 32, 64]
    model_cfg["fourier_init_hw"] = [2, 2]
    return model_cfg


def _run_tiny_pipeline(name: str, tmp_path: Path) -> None:
    model = build_model(name, _small_config(name))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(TinyDataset(), batch_size=2)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=BCEDiceLoss(),
        device="cpu",
        mixed_precision=False,
        use_aux_outputs_loss=False,
        use_boundary_loss=False,
    )
    train_metrics = trainer.train_one_epoch(loader, epoch=1)
    val_metrics = trainer.validate(loader)
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert 0.0 <= val_metrics["dice"] <= 1.0
    checkpoint = tmp_path / f"{name}.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = build_model(name, _small_config(name))
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=True)
    metrics = Evaluator(device="cpu", threshold=0.5, loss_fn=BCEDiceLoss()).evaluate(restored, loader)
    assert {"loss", "dice", "iou", "precision", "recall", "mae"} <= set(metrics)


def test_dapr_pipeline(tmp_path: Path) -> None:
    _run_tiny_pipeline("dapr_unet", tmp_path)


def test_dapr_residual_control_pipeline(tmp_path: Path) -> None:
    _run_tiny_pipeline("dapr_residual_control", tmp_path)
