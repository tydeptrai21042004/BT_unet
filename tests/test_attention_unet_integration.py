from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.output_utils import compute_supervised_loss
from src.losses import BCEDiceLoss
from src.models import build_model
from src.models.baselines.attention_unet import AttentionGate, AttentionUNetDecoderBlock
from src.models.baselines.resunetpp import ResUNetPlusPlus


def _cfg(name: str, directory: str = "configs/fair") -> dict:
    return yaml.safe_load((PROJECT_ROOT / directory / f"{name}.yaml").read_text(encoding="utf-8"))


def test_attention_unet_is_registered_configured_and_runs_forward_backward():
    torch.manual_seed(7)
    cfg = _cfg("attention_unet")
    cfg["model"]["channels"] = [2, 4, 8, 16, 32]
    model = build_model("attention_unet", cfg["model"]).train()

    x = torch.randn(2, 3, 32, 32)
    masks = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.float32)
    out = model(x)
    assert out.shape == masks.shape
    assert torch.isfinite(out).all()

    loss, logs, parsed = compute_supervised_loss(out, masks, main_loss_fn=BCEDiceLoss())
    assert parsed.main.shape == masks.shape
    assert logs["loss"] == float(loss.detach().item())
    loss.backward()

    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )


def test_attention_unet_uses_four_attention_gated_decoder_skips():
    model = build_model(
        "attention_unet",
        {
            "in_channels": 3,
            "num_classes": 1,
            "channels": [8, 16, 32, 64, 128],
            "norm": "bn",
            "act": "relu",
        },
    ).eval()
    blocks = [m for m in model.modules() if isinstance(m, AttentionUNetDecoderBlock)]
    gates = [m for m in model.modules() if isinstance(m, AttentionGate)]
    assert len(blocks) == 4
    assert len(gates) == 4


def test_attention_gate_is_conditioned_by_decoder_gate_signal():
    torch.manual_seed(11)
    gate = AttentionGate(skip_channels=4, gate_channels=8, inter_channels=2).eval()
    skip = torch.randn(1, 4, 16, 16)
    gate_low = torch.zeros(1, 8, 8, 8)
    gate_high = torch.ones(1, 8, 8, 8) * 3.0

    with torch.no_grad():
        out_low = gate(skip, gate_low)
        out_high = gate(skip, gate_high)

    assert out_low.shape == skip.shape
    assert out_high.shape == skip.shape
    assert torch.isfinite(out_low).all()
    assert torch.isfinite(out_high).all()
    assert not torch.allclose(out_low, out_high), "attention gate must depend on the decoder gating signal"


def test_resunetpp_contract_aliases_and_project_conv_are_available():
    model = ResUNetPlusPlus(channels=(8, 16, 32, 64, 128)).train()
    assert model.bridge is model.aspp_bridge
    assert model.dec3 is model.dec4
    assert hasattr(model.aspp_bridge.project, "conv")

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    loss = y.mean()
    loss.backward()
    assert model.aspp_bridge.project.conv.weight.grad is not None
    assert torch.isfinite(model.aspp_bridge.project.conv.weight.grad).all()
