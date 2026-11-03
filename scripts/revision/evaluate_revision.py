#!/usr/bin/env python3
"""Per-image Dice/IoU/boundary metrics, paired masks, corruption and probabilities.

External-domain targets must be withheld from training/validation and model selection.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common import ROOT, load_checkpoint, predict_probs, sha256_file, write_csv, write_json
from audit_splits import manifest_path
from src.analysis.boundary_metrics import measure_binary
from src.analysis.frequency_interventions import suppress_band, BANDS
from src.analysis.corruptions import CorruptThenTransform, SUPPORTED
from src.datasets import build_dataset, build_eval_transforms


def run(args):
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, name = load_checkpoint(args.checkpoint, args.model, args.config, device)
    training_seed = cfg.get('train', {}).get('seed')
    if training_seed is None:
        raise ValueError('Checkpoint config lacks the independent training seed; cannot run paired multi-seed analysis')
    img_size = int(args.image_size or cfg.get('data', {}).get('image_size', 352))
    transform = CorruptThenTransform(build_eval_transforms(image_size=img_size), args.corruption, args.severity, args.seed)
    manifest_path(args.data_root, args.dataset, args.split)
    dataset = build_dataset(name=args.dataset, root=args.data_root, split=args.split,
                            image_size=img_size, transform=transform, return_paths=True)
    if len(dataset) == 0:
        raise ValueError('Empty evaluation dataset')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.startswith('cuda'))
    cluster_by_id = {}
    if args.cluster_map:
        with Path(args.cluster_map).open(newline='', encoding='utf-8') as file:
            records = list(csv.DictReader(file))
        if not records or not {'sample_id', 'cluster_id'}.issubset(records[0]):
            raise ValueError('cluster-map must be CSV with sample_id,cluster_id columns')
        for item in records:
            sid = item['sample_id'].strip()
            group = item['cluster_id'].strip()
            if not sid or not group or sid in cluster_by_id:
                raise ValueError('Cluster map contains missing/duplicate ID or missing cluster')
            cluster_by_id[sid] = group
    rows = []
    output = Path(args.output_dir)
    intervention = suppress_band(model,args.remove_frequency_band) if args.remove_frequency_band else nullcontext()
    with torch.inference_mode(), intervention:
        for batch in loader:
            probs = predict_probs(model, batch['image'].to(device)).detach().float().cpu().numpy()
            masks = batch['mask'].detach().float().cpu().numpy()
            for i, sid in enumerate(batch['id']):
                sid = str(sid)
                values = measure_binary(probs[i, 0], masks[i, 0], args.threshold, args.boundary_tolerance)
                if args.cluster_map and sid not in cluster_by_id:
                    raise ValueError(f'Missing group label for evaluation sample {sid}')
                rows.append({'dataset': args.dataset, 'model': name, 'sample_id': sid,
                             'cluster_id': cluster_by_id.get(sid, ''),
                             'seed': int(training_seed), 'corruption_seed': args.seed, 'split': args.split, 'corruption': args.corruption,
                             'severity': args.severity, 'removed_frequency_band': args.remove_frequency_band or 'none', **values})
                if args.save_predictions:
                    for label, array in [('probabilities', probs[i,0]), ('masks', probs[i,0] >= args.threshold)]:
                        path = output / label / f'{sid}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.uint16(np.clip(array,0,1)*65535) if label == 'probabilities'
                                        else np.uint8(array)*255).save(path)
    write_csv(rows, output / 'per_image.csv')
    summary = {'checkpoint_sha256': sha256_file(args.checkpoint), 'model': name, 'dataset': args.dataset,
               'split': args.split, 'num_samples': len(dataset), 'corruption': args.corruption,
               'severity': args.severity, 'training_seed': int(training_seed), 'corruption_seed': args.seed,
               'group_map': str(args.cluster_map) if args.cluster_map else None,
               'removed_frequency_band': args.remove_frequency_band or 'none', 'threshold': args.threshold,
               'input_resolution': img_size, 'boundary_tolerance_pixels': args.boundary_tolerance,
               'distance_undefined_count': sum(not bool(r['distance_defined']) for r in rows),
               'complete_miss_count': sum(int(r['complete_miss']) for r in rows),
               'empty_prediction_count': sum(int(r['empty_prediction']) for r in rows),
               'metrics_mean': {key: float(np.nanmean([r[key] for r in rows]))
                                for key in ('dice','iou','precision','recall','mae','boundary_f1','hd95_pixels','assd_pixels')
                                if np.isfinite([r[key] for r in rows]).any()}}
    write_json(summary, output / 'summary.json')
    print('Wrote', output / 'per_image.csv', output / 'summary.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', default=None)
    p.add_argument('--config', default=None)
    p.add_argument('--dataset', required=True)
    p.add_argument('--data-root', default='data')
    p.add_argument('--cluster-map', help='Optional CSV: sample_id,cluster_id (patient/video grouping)')
    p.add_argument('--split', default='test', choices=['train','val','test'])
    p.add_argument('--image-size', type=int)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device', default='auto')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--boundary-tolerance', type=int, default=2)
    p.add_argument('--seed', type=int, default=42, help='Corruption RNG only. Model training seed is read from checkpoint.')
    p.add_argument('--corruption', choices=SUPPORTED, default='clean')
    p.add_argument('--remove-frequency-band', choices=BANDS.keys(), default=None,
                   help='BT-only inference intervention; records removed band per image.')
    p.add_argument('--severity', type=int, default=1)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--output-dir', required=True)
    run(p.parse_args())
