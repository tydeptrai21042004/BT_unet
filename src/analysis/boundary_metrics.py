"""Binary mask metrics evaluated at a documented common pixel resolution.

Distances use pixels. Empty-mask conventions: both empty -> perfect overlap and
0 distance; one empty -> HD95/ASSD undefined (NaN), boundary F1 = 0.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _boundary(mask):
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def measure_binary(prob, target, threshold=0.5, boundary_tolerance=2):
    prob = np.asarray(prob, dtype=np.float64).squeeze()
    gt = np.asarray(target).squeeze() >= 0.5
    pred = prob >= threshold
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError('Expected same-size HxW prediction and target')
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    smooth = 1.0  # matches repository Dice implementation
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    mae = float(np.abs(prob - gt.astype(float)).mean())
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        bf1, hd95, assd = 1.0, 0.0, 0.0
    elif not bp.any() or not bg.any():
        bf1, hd95, assd = 0.0, float('nan'), float('nan')
    else:
        dist_to_gt = ndimage.distance_transform_edt(~bg)
        dist_to_pred = ndimage.distance_transform_edt(~bp)
        p2g = dist_to_gt[bp]
        g2p = dist_to_pred[bg]
        precision_b = float((p2g <= boundary_tolerance).mean())
        recall_b = float((g2p <= boundary_tolerance).mean())
        bf1 = 2 * precision_b * recall_b / max(precision_b + recall_b, 1e-12)
        hd95 = float(np.percentile(np.concatenate([p2g, g2p]), 95))
        assd = float((p2g.sum() + g2p.sum()) / (p2g.size + g2p.size))
    return dict(dice=float(dice), iou=float(iou),
                precision=float(tp / (tp + fp + 1e-7)),
                recall=float(tp / (tp + fn + 1e-7)),
                mae=mae, boundary_f1=float(bf1), hd95_pixels=hd95,
                assd_pixels=assd, lesion_fraction=float(gt.mean()))
