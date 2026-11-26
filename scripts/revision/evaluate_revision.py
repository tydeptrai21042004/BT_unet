#!/usr/bin/env python3
"""Per-image Dice/IoU/boundary metrics, paired masks, corruption and probabilities.

External-domain targets must be withheld from training/validation and model selection.
"""
from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common import ROOT, load_checkpoint, sha256_file, write_csv, write_json
from src.engine.output_utils import parse_model_output
from scripts.train_one import build_loss
from audit_splits import manifest_path
from src.analysis.boundary_metrics import measure_binary
from src.analysis.extended_metrics import probability_metrics, confusion_metrics
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
    evaluation_start = time.perf_counter()
    is_cuda = str(device).startswith("cuda")
    if is_cuda: torch.cuda.reset_peak_memory_stats(device)
    forward_times = []
    # The validation evaluator measures the configured main-head loss averaged
    # over batches. Use that identical definition for the held-out test split;
    # validation loss must never be substituted for test loss.
    loss_name = str(cfg.get('train', {}).get('loss', 'bce_dice'))
    loss_fn = build_loss(loss_name).to(device).eval()
    loss_sum = 0.0
    loss_batches = 0
    output = Path(args.output_dir)
    intervention = suppress_band(model,args.remove_frequency_band) if args.remove_frequency_band else nullcontext()
    with torch.inference_mode(), intervention:
        for batch in loader:
            images = batch['image'].to(device)
            masks_device = batch['mask'].to(device)
            if is_cuda: torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits = parse_model_output(model(images)).main
            if is_cuda: torch.cuda.synchronize(device)
            forward_times.append((time.perf_counter()-t0, len(images)))
            batch_loss = float(loss_fn(logits, masks_device).item())
            if not np.isfinite(batch_loss):
                raise ValueError('Non-finite loss on evaluation batch')
            loss_sum += batch_loss
            loss_batches += 1
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            masks = masks_device.detach().float().cpu().numpy()
            for i, sid in enumerate(batch['id']):
                sid = str(sid)
                values = measure_binary(probs[i, 0], masks[i, 0], args.threshold, args.boundary_tolerance)
                if args.cluster_map and sid not in cluster_by_id:
                    raise ValueError(f'Missing group label for evaluation sample {sid}')
                binary = probs[i, 0] >= args.threshold
                truth = masks[i, 0] >= .5
                tp = int(np.logical_and(binary, truth).sum())
                fp = int(np.logical_and(binary, ~truth).sum())
                tn = int(np.logical_and(~binary, ~truth).sum())
                fn = int(np.logical_and(~binary, truth).sum())
                accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
                specificity = tn / max(tn + fp, 1)
                balanced_accuracy = .5 * (tp / max(tp + fn, 1) + specificity)
                f2 = 5 * tp / max(5 * tp + 4 * fn + fp, 1)
                values.update(pixel_accuracy=float(accuracy), specificity=float(specificity),
                              balanced_accuracy=float(balanced_accuracy), f2_score=float(f2))
                values.update(confusion_metrics(tp, fp, tn, fn))
                values.update(probability_metrics(probs[i,0], truth))
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
    metric_keys = ('dice', 'iou', 'precision', 'recall', 'mae',
                   'boundary_f1', 'hd95_pixels', 'assd_pixels',
                   'pixel_accuracy', 'specificity', 'balanced_accuracy', 'f2_score',
                   'surface_dice','mcc','foreground_area_error','brier','nll','ece','foreground_prediction_ece','auprc')
    metric_mean = {}
    metric_std = {}
    metric_median = {}
    metric_defined_count = {}
    for key in metric_keys:
        values = np.asarray([r[key] for r in rows], dtype=float)
        good = values[np.isfinite(values)]
        metric_defined_count[key] = len(good)
        if len(good):
            metric_mean[key] = float(good.mean())
            metric_std[key] = float(good.std(ddof=1)) if len(good) > 1 else 0.0
            metric_median[key] = float(np.median(good))
    test_loss = float(loss_sum / loss_batches)
    metric_mean['loss'] = test_loss
    positives = sum(not bool(r['empty_ground_truth']) for r in rows)
    if positives:
        metric_mean['complete_miss_rate'] = sum(r['complete_miss'] for r in rows)/positives
    metric_mean['empty_prediction_rate'] = sum(r['empty_prediction'] for r in rows)/len(rows)
    metric_mean['undefined_distance_rate'] = sum(not bool(r['distance_defined']) for r in rows)/len(rows)
    summary = {'checkpoint_sha256': sha256_file(args.checkpoint), 'model': name, 'dataset': args.dataset,
               'split': args.split, 'num_samples': len(dataset), 'corruption': args.corruption,
               'severity': args.severity, 'training_seed': int(training_seed), 'corruption_seed': args.seed,
               'group_map': str(args.cluster_map) if args.cluster_map else None,
               'removed_frequency_band': args.remove_frequency_band or 'none', 'threshold': args.threshold,
               'input_resolution': img_size, 'boundary_tolerance_pixels': args.boundary_tolerance,
               'test_loss': test_loss, 'loss_name': loss_name,
               'loss_protocol': 'configured main-head loss, arithmetic mean of evaluation batch losses (same as validation)',
               'test_loss_num_batches': loss_batches,
               'evaluation_seconds': time.perf_counter()-evaluation_start,
               'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'parameters_total':sum(p.numel() for p in model.parameters()),
               'eval_forward_seconds':sum(t for t,n in forward_times),
               'eval_forward_images_per_second':sum(n for t,n in forward_times)/max(sum(t for t,n in forward_times),1e-12),
               'eval_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/1048576 if is_cuda else None,
               'eval_timing_note':'Actual test batches incl. first forward; not warmup-controlled batch-1 latency. See benchmark_inference for standardized timing.',
               'positive_sample_count': positives,
               'metric_protocol_extended': 'macro per-image, surface dice tolerance equals boundary tolerance; AP is tie-aware; MCC undefined when denominator zero; foreground prediction ECE bins only p>=0.5',
               'distance_undefined_count': sum(not bool(r['distance_defined']) for r in rows),
               'complete_miss_count': sum(int(r['complete_miss']) for r in rows),
               'empty_prediction_count': sum(int(r['empty_prediction']) for r in rows),
               'metrics_mean': metric_mean,
               'metrics_std_across_images': metric_std,
               'metrics_median_across_images': metric_median,
               'metrics_defined_count': metric_defined_count}
    write_json(summary, output / 'summary.json')
    print('[TEST METRICS]', f'dataset={args.dataset}', f'model={name}', f'seed={training_seed}',
          ' '.join(f'{k}={metric_mean[k]:.6f}' for k in ('dice','iou','precision','recall','mae','loss')),
          flush=True)
    print('[TEST EXTRA]',
          ' '.join(f'{k}={metric_mean[k]:.6f}' for k in ('boundary_f1','hd95_pixels','assd_pixels',
                  'pixel_accuracy','specificity','balanced_accuracy','f2_score') if k in metric_mean),
          flush=True)
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
