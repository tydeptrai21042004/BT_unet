"""Inference-only BT spectral frequency-band interventions.

Bottleneck weights are NOT modified. Suppression happens immediately before the
shared complex channel mixer, on actual rfft2 frequency coordinates. The
resulting intervention is a sensitivity test, not a causal proof of clinical
artifact suppression.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

BANDS = {'low': (0.0, .18), 'middle': (.18, .35), 'high': (.35, .71)}


def spectral_band_mask(shape, height, width, band, device):
    if band not in BANDS:
        raise ValueError(f'Unknown frequency band: {band}')
    fy=torch.fft.fftfreq(height,device=device).reshape(height,1)
    fx=torch.fft.rfftfreq(width,device=device).reshape(1,-1)
    radius=torch.sqrt(fy.square()+fx.square())
    if tuple(radius.shape)!=tuple(shape):
        raise ValueError(f'Frequency grid shape {radius.shape} != spectrum {shape}')
    lo,hi=BANDS[band]
    return (radius>=lo)&(radius < hi if band!='high' else radius<=hi)


@contextmanager
def suppress_band(model, band):
    """Patches only an instance's spectral mixer while inside this context."""
    if not hasattr(model,'fourier_bottleneck'):
        raise ValueError('Frequency intervention requires BT-style Fourier bottleneck')
    bt=model.fourier_bottleneck
    old_mix=bt._mix
    current_size=[]

    def cache_size(module, inputs):
        current_size[:] = list(inputs[0].shape[-2:])

    def masked_mix(spectrum):
        if not current_size:raise RuntimeError('BT spatial size not captured')
        h,w=current_size
        drop=spectral_band_mask(spectrum.shape[-2:],h,w,band,spectrum.device)
        return old_mix(spectrum*(~drop).to(dtype=spectrum.real.dtype)[None,None])

    hook=bt.register_forward_pre_hook(cache_size)
    bt._mix=masked_mix
    try:
        yield model
    finally:
        bt._mix=old_mix
        hook.remove()
