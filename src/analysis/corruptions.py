"""Deterministic geometry-preserving image-only corruptions; masks are unchanged."""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SUPPORTED = ('clean', 'gaussian_noise', 'gaussian_blur', 'motion_blur', 'low_contrast', 'dark', 'jpeg', 'glare')


def corrupt(image: Image.Image, kind='clean', severity=1, seed=42, sample_id=''):
    if kind not in SUPPORTED:
        raise ValueError(f'Unknown corruption {kind}; choose {SUPPORTED}')
    if not 1 <= int(severity) <= 4:
        raise ValueError('Severity must be in [1,4]')
    if kind == 'clean':
        return image.copy()
    digest = hashlib.sha256(f'{seed}:{sample_id}:{kind}:{severity}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    severity = int(severity)
    if kind == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur(radius=severity * 0.9))
    if kind == 'low_contrast':
        return ImageEnhance.Contrast(image).enhance(max(0.15, 1 - severity * 0.2))
    if kind == 'dark':
        return ImageEnhance.Brightness(image).enhance(max(0.2, 1 - severity * 0.18))
    if kind == 'jpeg':
        stream = io.BytesIO()
        image.convert('RGB').save(stream, format='JPEG', quality=max(8, 70 - severity * 16))
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert('RGB')
    data = np.asarray(image.convert('RGB')).astype(np.float32)
    if kind == 'gaussian_noise':
        data += rng.normal(0, severity * 9, data.shape)
    elif kind == 'motion_blur':
        # Horizontal box convolution with reflect padding; deterministic direction.
        radius = severity * 2 + 1
        from scipy.ndimage import uniform_filter1d
        data = uniform_filter1d(data, size=radius, axis=1, mode='reflect')
    elif kind == 'glare':
        h, w = data.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx, cy = int(rng.uniform(.2, .8) * w), int(rng.uniform(.2, .8) * h)
        radius = max(1, min(h, w) * (0.045 + severity * .015))
        alpha = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*radius**2))[..., None]
        data = data * (1 - alpha) + 255 * alpha
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), 'RGB')


class CorruptThenTransform:
    def __init__(self, transform, kind='clean', severity=1, seed=42):
        self.transform, self.kind, self.severity, self.seed = transform, kind, severity, seed

    def __call__(self, sample):
        sample['image'] = corrupt(sample['image'], self.kind, self.severity, self.seed, str(sample['id']))
        return self.transform(sample)
