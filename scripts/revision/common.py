"""Shared, read-only checkpoint loading for revision evaluations.

All script outputs are derived from checkpoints; no training results are fabricated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.engine.output_utils import parse_model_output
from src.utils import load_yaml


def config_from_checkpoint(path, fallback_config=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get('config'), dict):
        return ckpt['config'], ckpt
    if not fallback_config:
        raise ValueError('Checkpoint missing resolved config; supply --config')
    return load_yaml(fallback_config), ckpt


def load_checkpoint(path, model_name=None, config=None, device='cpu'):
    cfg, ckpt = config_from_checkpoint(path, config)
    model_cfg = cfg['model'].copy()
    name = model_name or model_cfg.pop('name')
    model_cfg.pop('name', None)
    model = build_model(name, config=model_cfg)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, cfg, name


def predict_probs(model, images):
    return torch.sigmoid(parse_model_output(model(images)).main)


def write_csv(rows, path, columns=None):
    if not rows:
        raise ValueError('No data rows to export')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
