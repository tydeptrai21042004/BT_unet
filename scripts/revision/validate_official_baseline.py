#!/usr/bin/env python3
"""Fail-closed validation of user-installed/pinned official spectral baselines.

This script does not download, recreate, or falsely certify GFUNet/FDFUNet/etc.
It verifies pinned source + audit via VerifiedExternalSpectral, checks forward,
backpropagation, trainable parameters and full input/output shape on real code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

try:
    from .common import write_json,sha256_file
except ImportError:
    from common import write_json,sha256_file
from src.models import build_model
from src.models.official.strict_external import verify_source


def validate(config,device='cpu',image_size=128):
    path=Path(config)
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    spec=cfg.get('model',{})
    if spec.get('name')!='verified_external_spectral':
        raise ValueError('Must use the original, audited verified_external_spectral adapter')
    audit=verify_source(spec['source_dir'],spec['expected_commit'],spec['audit_file'])
    model_cfg={k:v for k,v in spec.items() if k!='name'}
    model=build_model('verified_external_spectral',config=model_cfg).to(device)
    model.train()
    in_channels=int(cfg.get('data',{}).get('in_channels',3))
    example=torch.randn(1,in_channels,image_size,image_size,device=device)
    output=model(example)
    if output.shape!=(1,1,image_size,image_size):
        raise ValueError('Required full-resolution single-mask logits, got '+str(tuple(output.shape)))
    target=torch.randint(0,2,output.shape,device=device).float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(output,target)
    loss.backward()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters or not any(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError('Official model does not supply finite training gradients')
    return {'source_commit':audit['commit'],'source_repository':audit['official_repository'],
            'source_audit_file':str(spec['audit_file']), 'configuration_sha256':sha256_file(path),
            'model_class':type(model.model).__name__, 'validation':'passed_forward_and_backward',
            'output_shape':list(output.shape), 'trainable_parameters':sum(p.numel() for p in parameters),
            'warning':'Runtime contract passed; independent paper-fidelity verification and genuine benchmark results still required.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--image-size',type=int,default=128)
    a=p.parse_args();report=validate(a.config,a.device,a.image_size)
    write_json(report,a.output)
    print('PASS:',a.output)
