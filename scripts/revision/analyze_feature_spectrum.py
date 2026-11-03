#!/usr/bin/env python3
"""Activation-conditioned pre/post BT spectra, never synthetic signal results.

For the trained BT bottleneck, captures the projected pre-FFT tensor and the
post-IFFT tensor before dropout/output projection. Uses the same full FFT for
both real signals, avoiding rFFT half-spectrum energy weighting mistakes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .common import load_checkpoint, write_csv, write_json, sha256_file
    from .audit_splits import manifest_path
except ImportError:
    from common import load_checkpoint, write_csv, write_json, sha256_file
    from audit_splits import manifest_path
from src.datasets import build_dataset, build_eval_transforms

BANDS={'low':(0.,.18),'middle':(.18,.35),'high':(.35,2**-.5+1e-6)}


def band_fractions(x: torch.Tensor) -> dict[str,float]:
    if x.ndim != 4 or not torch.isfinite(x).all():
        raise ValueError('Expected finite projected BCHW real features')
    h,w=x.shape[-2:]
    spec=torch.fft.fft2(x.float(),norm='ortho')
    power=spec.abs().square().mean((0,1))
    fy=torch.fft.fftfreq(h,device=x.device)[:,None]
    fx=torch.fft.fftfreq(w,device=x.device)[None,:]
    rad=(fy.square()+fx.square()).sqrt()
    total=power.sum().clamp_min(1e-20)
    out={}
    for label,(lo,hi) in BANDS.items():
        selected=(rad>=lo)&(rad<hi)
        out[label]=float((power[selected].sum()/total).item())
    out['energy_per_pixel']=float(power.mean().item())
    return out


def analyze(checkpoint,dataset,data_root,output,split='test',device='auto',
            image_size=None,max_images=100,num_workers=0):
    actual_device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    model,cfg,name=load_checkpoint(checkpoint,device=actual_device)
    if not hasattr(model,'fourier_bottleneck'):
        raise ValueError('Requires model with Fourier bottleneck')
    manifest_path(data_root,dataset,split)
    resolution=image_size or cfg.get('data',{}).get('image_size',352)
    data=build_dataset(name=dataset,root=data_root,split=split,image_size=int(resolution),
                       transform=build_eval_transforms(int(resolution)),return_paths=True)
    loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=num_workers)
    observed={}
    bt=model.fourier_bottleneck
    def capture_before(_module,_inputs,output): observed['before']=output.detach()
    def capture_after(_module,inputs): observed['after']=inputs[0].detach()
    handles=[bt.pre_act.register_forward_hook(capture_before),
             bt.dropout.register_forward_pre_hook(capture_after)]
    rows=[]
    try:
        with torch.inference_mode():
            for batch in loader:
                if len(rows)>=max_images:
                    break
                observed.clear()
                _=model(batch['image'].to(actual_device))
                if set(observed)!={'before','after'}:
                    raise RuntimeError('Did not capture both spectral activations')
                a=band_fractions(observed['before']);b=band_fractions(observed['after'])
                row={'sample_id':str(batch['id'][0]), 'train_seed':int(cfg['train']['seed'])}
                for key in a:
                    row[f'before_{key}']=a[key];row[f'after_{key}']=b[key]
                    row[f'delta_{key}']=b[key]-a[key]
                rows.append(row)
    finally:
        for handle in handles:handle.remove()
    if not rows:raise ValueError('No images analyzed')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_csv(rows,output/'activation_spectral_energy.csv')
    write_json({'checkpoint_sha256':sha256_file(checkpoint),'dataset':dataset,
                'split':split,'image_count':len(rows),'bands':BANDS,
                'note':'Frequency energy proportions are measured from actual projected/reconstructed bottleneck features, not raw images or clinical denoising.'},
               output/'activation_spectral_metadata.json')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',default='data');p.add_argument('--split',default='test')
    p.add_argument('--image-size',type=int);p.add_argument('--max-images',type=int,default=100)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--device',default='auto');p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    analyze(a.checkpoint,a.dataset,a.data_root,a.output_dir,a.split,a.device,a.image_size,a.max_images,a.num_workers)
