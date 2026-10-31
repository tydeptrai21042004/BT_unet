#!/usr/bin/env python3
"""Checkpoint-grounded BT frequency-response heatmaps and signal-energy analysis.

Charts describe learned operators; they do NOT establish causal denoising claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_checkpoint, write_csv, write_json, sha256_file


def radial_profile(values, bins=12):
    """Half-spectrum frequency radius in normalized cycles/pixel."""
    h, w = values.shape
    # rfft2 stores complete rows and nonnegative frequencies on last axis.
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.linspace(0, 0.5, w, dtype=float)[None, :] if w > 1 else np.zeros((1, 1))
    r = np.sqrt(fy**2 + fx**2)
    edges = np.linspace(0, np.sqrt(.5), bins + 1)
    result = []
    for i in range(bins):
        select = (r >= edges[i]) & ((r <= edges[i+1]) if i == bins - 1 else (r < edges[i+1]))
        result.append({'freq_min': edges[i], 'freq_max': edges[i+1], 'mean': float(np.mean(values[select])) if np.any(select) else float('nan')})
    return result


def analyze_model(model, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bt = model.fourier_bottleneck
    shape = (int(bt.amplitude_logits.shape[-2]), int(bt.amplitude_logits.shape[-1]))
    with torch.no_grad():
        # Generated at learned base grid, not necessarily input-dependent interpolated grid.
        dummy = torch.zeros((1, bt.hidden_channels, *shape), device=next(bt.parameters()).device, dtype=torch.complex64)
        gain, phase = bt._response(dummy)
        gain, phase = gain.cpu().numpy()[0], phase.cpu().numpy()[0]
        if bt.use_channel_mixing:
            m = bt.spectral_channel_mixer.weight.detach().cpu().numpy()[:, :, 0, 0]
        else:
            m = np.eye(bt.hidden_channels)
    # Exact Frobenius norm of M diag(A exp(iPhi)) at each stored frequency.
    column_norm = np.sum(m.astype(np.float64)**2, axis=0)
    fro = np.sqrt(np.tensordot(column_norm, gain.astype(np.float64)**2, axes=(0,0)))
    amplitude_mean = gain.mean(axis=0)
    # Mean complex unit vector magnitude avoids misleading linear phase average.
    phase_consistency = np.abs(np.exp(1j * phase).mean(axis=0))
    phase_mean = np.angle(np.exp(1j * phase).mean(axis=0))
    mats = [('mean_amplitude', amplitude_mean), ('mean_circular_phase', phase_mean),
            ('phase_concentration', phase_consistency), ('spectral_core_frobenius_gain', fro)]
    output.mkdir(parents=True, exist_ok=True)
    for name, array in mats:
        np.save(output / f'{name}.npy', array)
    for name, arr in mats:
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(np.fft.fftshift(arr, axes=0), aspect='auto', interpolation='nearest')
        ax.set(xlabel='Nonnegative horizontal rFFT frequency bin', ylabel='Centered vertical frequency bin', title=name.replace('_',' '))
        fig.colorbar(im, ax=ax)
        fig.tight_layout();fig.savefig(output / f'{name}.png', dpi=180);plt.close(fig)
    rows = []
    for label, data in [('amplitude', amplitude_mean), ('operator_frobenius', fro)]:
        for index, row in enumerate(radial_profile(data)):
            rows.append({'quantity': label, 'bin': index, **row})
    write_csv(rows, output / 'radial_profiles.csv')
    write_json({'hidden_channels': bt.hidden_channels, 'grid': shape,
                'amplitude_gain_min': float(gain.min()), 'amplitude_gain_mean':float(gain.mean()),
                'amplitude_gain_max':float(gain.max()), 'phase_abs_mean':float(np.abs(phase).mean()),
                'note': 'Spectral core Frobenius norm is pre-irfft, not a learned low/high-pass classification.'},
               output / 'spectrum_summary.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model, cfg, name = load_checkpoint(args.checkpoint,config=args.config,device=args.device)
    if not hasattr(model,'fourier_bottleneck'):
        raise ValueError(f'{name} has no Fourier bottleneck')
    analyze_model(model,Path(args.output_dir))
    print('Saved actual learned BT spectral diagnostics')
