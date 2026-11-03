"""Strict interface to an UNMODIFIED upstream segmentation network.

A configured baseline is not an implementation of GFUNet, AFDSeg or any other
paper. The authentic code MUST be installed separately and audited. This module
intentionally refuses to call the model unless a pinned source and a complete
user-supplied audit record are available. It never substitutes a toy FFT block.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..registry import register_model


def git_commit(source_dir: str | Path) -> str:
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Official source not installed: {root}')
    proc = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f'Official source is not a Git checkout: {root}. {proc.stderr}')
    return proc.stdout.strip().lower()


def verify_source(source_dir: str | Path, expected_commit: str, audit_file: str | Path) -> dict:
    root = Path(source_dir).expanduser().resolve()
    commit = git_commit(root)
    if not expected_commit or len(expected_commit) < 7 or not commit.startswith(expected_commit.lower()):
        raise ValueError(f'Official source commit mismatch: expected {expected_commit!r}, actual {commit}')
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                            capture_output=True, text=True, check=True)
    if status.stdout.strip():
        raise ValueError('Original baseline source has uncommitted edits; audit a clean pinned checkout')
    audit_path = Path(audit_file).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f'Missing paper-to-code audit: {audit_path}')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('status') != 'verified':
        raise ValueError('Baseline has not passed paper-to-code audit; status must be verified')
    if not audit.get('official_repository') or not audit.get('paper_reference'):
        raise ValueError('Audit requires official_repository and paper_reference')
    if audit.get('commit', '').lower() != commit:
        raise ValueError('Audit commit must equal the complete checked-out source SHA')
    components = audit.get('component_checks')
    if not isinstance(components, dict) or len(components) < 3 or not all(
        isinstance(v, str) and v.strip() for v in components.values()
    ):
        raise ValueError('Audit component_checks needs at least three documented paper-to-source mappings')
    for name, source_ref in components.items():
        source_file = source_ref.split('::', 1)[0]
        path = (root / source_file).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Audit component {name!r} does not reference a real original source file: {source_ref}')
    return audit


def _resolve_primary(raw: Any, key: str | None, index: int | None):
    if key is not None:
        if not isinstance(raw, dict) or key not in raw:
            raise TypeError(f'Upstream output must have dictionary key {key!r}')
        selected = raw[key]
    elif index is not None:
        if not isinstance(raw, (list, tuple)) or index >= len(raw) or index < -len(raw):
            raise TypeError(f'Upstream output must have valid index {index!r}')
        selected = raw[index]
    else:
        if not torch.is_tensor(raw):
            raise TypeError('Upstream outputs multiple tensors; configure explicit output_key or output_index')
        selected = raw
    if not torch.is_tensor(selected):
        raise TypeError('Selected upstream prediction is not a tensor')
    if selected.ndim != 4:
        raise ValueError(f'Expected BCHW prediction, got shape {tuple(selected.shape)}')
    if not torch.isfinite(selected).all():
        raise ValueError('Upstream prediction contains nonfinite values')
    return selected


@register_model('verified_external_spectral')
class VerifiedExternalSpectral(nn.Module):
    """Only wire original modules, with explicit output selection/conversion.

    This adapter does not change the upstream forward graph. Output conversions
    are explicitly recorded; logits mode is preferred for the shared BCE/Dice
    trainer. The faithful upstream training protocol must be run separately when
    it requires custom losses, pretraining or specialized dataset handling.
    """
    def __init__(self, *, source_dir: str, expected_commit: str, audit_file: str,
                 module_path: str, entrypoint: str, model_kwargs: dict | None = None,
                 output_key: str | None = None, output_index: int | None = None,
                 output_type: str = 'logits', output_channels: int = 1,
                 allow_resize: bool = False) -> None:
        super().__init__()
        self.audit = verify_source(source_dir, expected_commit, audit_file)
        if not module_path or not entrypoint:
            raise ValueError('module_path and entrypoint are required')
        root = Path(source_dir).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        upstream = importlib.import_module(module_path)
        imported_file = Path(getattr(upstream, '__file__', '')).resolve()
        if not imported_file.is_relative_to(root):
            raise RuntimeError(f'Imported module {module_path!r} is not from pinned official checkout: {imported_file}')
        factory = getattr(upstream, entrypoint)
        model = factory(**(model_kwargs or {}))
        if not isinstance(model, nn.Module):
            raise TypeError('Official entrypoint did not construct a torch.nn.Module')
        if output_key is not None and output_index is not None:
            raise ValueError('Specify output_key OR output_index')
        if output_type not in {'logits', 'probabilities'}:
            raise ValueError('Declare whether upstream output contains logits or probabilities')
        if int(output_channels) <= 0:
            raise ValueError('output_channels must be positive')
        self.model = model
        self.output_key = output_key
        self.output_index = output_index
        self.output_type = output_type
        self.output_channels = int(output_channels)
        self.allow_resize = bool(allow_resize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = _resolve_primary(self.model(x), self.output_key, self.output_index)
        if pred.size(0) != x.size(0) or pred.size(1) != self.output_channels:
            raise ValueError(f'Official output batch/channel mismatch: {tuple(pred.shape)}')
        if pred.shape[-2:] != x.shape[-2:]:
            if not self.allow_resize:
                raise ValueError('Upstream output spatial size differs; explicit allow_resize required')
            pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if self.output_type == 'probabilities':
            if pred.detach().amin().item() < -1e-5 or pred.detach().amax().item() > 1 + 1e-5:
                raise ValueError('Declared probabilities but upstream predictions are outside [0,1]')
            pred = torch.logit(pred.clamp(1e-6, 1-1e-6))
        return pred
