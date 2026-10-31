"""Opt-in integration adapter for authentic third-party spectral implementations.

This adapter does not reproduce or claim to be any named published architecture.
Users must install an audited original implementation on PYTHONPATH; outputs
must be compatible with the repository's logits contract.
"""
from __future__ import annotations

from importlib import import_module

import torch.nn as nn

from ..registry import register_model


@register_model('external_spectral')
class ExternalSpectralBaseline(nn.Module):
    def __init__(self, module_path: str, class_name: str, model_kwargs: dict | None = None) -> None:
        super().__init__()
        if not module_path or not class_name:
            raise ValueError('Both module_path and class_name are required')
        imported=import_module(module_path)
        cls=getattr(imported,class_name)
        model=cls(**(model_kwargs or {}))
        if not isinstance(model,nn.Module):
            raise TypeError('Imported baseline must subclass torch.nn.Module')
        self.model=model

    def forward(self, x):
        return self.model(x)
