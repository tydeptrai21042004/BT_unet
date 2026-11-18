#!/usr/bin/env python3
"""Small, verifiable AUTO results for the 165-run manuscript table.

Only completed, clean, held-out evaluations are exported as quantitative rows.
No weights, prediction PNGs or nested ZIPs are included.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

MAIN = ('dice', 'iou', 'precision', 'recall', 'mae', 'loss')
EXTRA = ('boundary_f1', 'hd95_pixels', 'assd_pixels', 'pixel_accuracy',
         'specificity', 'balanced_accuracy', 'f2_score')
COLUMNS = ('experiment_number', 'experiment_id', 'dataset', 'model', 'variant',
           'seed', 'epochs', 'status') + MAIN + EXTRA


def _write_csv(path, records, cols=COLUMNS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        out = csv.DictWriter(handle, fieldnames=cols, extrasaction='ignore')
        out.writeheader()
        out.writerows(records)


def _metric(summary, name):
    metrics = summary.get('metrics_mean', {})
    v = metrics.get(name, summary.get('test_loss') if name == 'loss' else None)
    if v is None or not math.isfinite(float(v)):
        return '---'
    return f'{float(v):.6f}'


def _record(index, spec, summary):
    # An experiment must not appear completed without a real, independently
    # evaluated test split and every metric needed in the manuscript.
    for key, expected in [('dataset', spec['dataset']),
                          ('training_seed', int(spec['seed'])),
                          ('split', 'test'), ('corruption', 'clean')]:
        if summary.get(key) != expected:
            raise ValueError(f'{spec["id"]}: evaluation {key} mismatch: {summary.get(key)!r} != {expected!r}')
    if summary.get('removed_frequency_band', 'none') != 'none':
        raise ValueError('Manuscript row cannot use frequency-intervention results')
    if not summary.get('checkpoint_sha256') or not summary.get('num_samples', 0):
        raise ValueError('Evaluation is missing checkpoint hash or test sample count')
    out = {'experiment_number': index, 'experiment_id': spec['id'],
           'dataset': spec['dataset'], 'model': spec['model'],
           'variant': spec.get('variant', ''), 'seed': spec['seed'],
           'epochs': spec['epochs'], 'status': 'complete'}
    out.update({key: _metric(summary, key) for key in MAIN + EXTRA})
    if any(out[key] == '---' for key in MAIN):
        raise ValueError(f'{spec["id"]}: incomplete six-paper-metric evaluation')
    return out


def _extract_eval_output(command):
    tokens = list(map(str, command))
    if '--output-dir' not in tokens:
        raise ValueError('Evaluator command missing --output-dir')
    i = tokens.index('--output-dir')
    return Path(tokens[i + 1])


def export_completed(index, spec, evaluations, results_root, compact_root, metadata):
    """Copy one finished experiment's metrics; never store model weights."""
    experiment_dir = Path(compact_root) / f'{index:04d}'
    if experiment_dir.exists():
        raise FileExistsError(f'Already exported completed experiment: {experiment_dir}')
    # Validate *all* requested test targets before committing the export.
    collected = []
    for label, command in evaluations:
        folder = _extract_eval_output(command)
        summary_file = folder / 'summary.json'
        per_image_file = folder / 'per_image.csv'
        if not summary_file.is_file() or not per_image_file.is_file():
            raise FileNotFoundError(f'Evaluation missing summary or per_image CSV: {folder}')
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        if summary.get('dataset') != label:
            raise ValueError(f'{spec["id"]}: target label mismatch')
        collected.append((label, summary, summary_file, per_image_file))
    if not collected:
        raise ValueError('No evaluated targets')
    chosen = next((data for data in collected if data[0] == spec['dataset']), collected[0])
    primary_row = _record(index, spec, chosen[1])

    experiment_dir.mkdir(parents=True)
    for label, summary, summary_file, per_image_file in collected:
        target = experiment_dir / 'evaluations' / label
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary_file, target / 'summary.json')
        shutil.copyfile(per_image_file, target / 'per_image.csv')
    (experiment_dir / 'paper_row.json').write_text(json.dumps(primary_row, indent=2) + '\n', encoding='utf-8')
    (experiment_dir / 'provenance.json').write_text(json.dumps({
        'experiment_number': index, 'experiment': spec, 'training_run': metadata,
        'test_split': chosen[1]['split'],
        'test_sample_count': chosen[1]['num_samples'],
        'checkpoint_sha256': chosen[1]['checkpoint_sha256'],
    }, indent=2, default=str) + '\n', encoding='utf-8')

    results_root = Path(results_root)
    for name, pattern in [('training_summary.json', '**/results/summary.json'),
                          ('history.csv', '**/results/history.csv'),
                          ('resolved_train_config.yaml', '**/resolved_train_config.yaml')]:
        sources = list(results_root.glob(pattern))
        if len(sources) == 1 and sources[0].stat().st_size < 10_000_000:
            shutil.copyfile(sources[0], experiment_dir / name)
    analysis = results_root / 'analysis'
    if analysis.is_dir():
        budget = 15_000_000  # optional posthoc summaries only
        for source in sorted(analysis.rglob('*')):
            if not source.is_file() or source.suffix not in {'.json', '.csv'}:
                continue
            size = source.stat().st_size
            if size > budget:
                continue
            dest = experiment_dir / 'analysis' / source.relative_to(analysis)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            budget -= size
    return primary_row, experiment_dir


def update_reports(base, compact_root, plan):
    """Create a 165-row CSV plus completed-only CSV, JSON and copy-ready LaTeX."""
    base, compact_root = Path(base), Path(compact_root)
    filled, complete = [], []
    for index, spec in enumerate(plan, 1):
        record = {'experiment_number': index, 'experiment_id': spec['id'],
                  'dataset': spec['dataset'], 'model': spec['model'],
                  'variant': spec.get('variant', ''), 'seed': spec['seed'],
                  'epochs': spec['epochs'], 'status': 'pending'}
        record.update({metric: '---' for metric in MAIN + EXTRA})
        found = compact_root / f'{index:04d}' / 'paper_row.json'
        if found.is_file():
            saved = json.loads(found.read_text(encoding='utf-8'))
            for field in ('experiment_number','experiment_id','dataset','seed','epochs'):
                if str(saved[field]) != str(record[field]):
                    raise ValueError(f'Completed result metadata mismatch in {found}: {field}')
            record.update(saved)
            complete.append(record)
        filled.append(record)
    suffix = 'AUTO'
    _write_csv(base / f'BT_UNET_PAPER_RESULTS_{suffix}.csv', filled)
    _write_csv(base / f'BT_UNET_COMPLETED_RESULTS_{suffix}.csv', complete)
    (base / f'BT_UNET_COMPLETED_RESULTS_{suffix}.json').write_text(
        json.dumps(complete, indent=2) + '\n', encoding='utf-8')
    # Exact six numeric cells from the user's 165-line paper table, preserving
    # all other identifying columns when the user pastes the replacement row.
    lines = ['% Completed rows only; replace the matching No. in your LaTeX longtable.']
    for row in complete:
        values = ' & '.join('\\(' + row[m] + '\\)' for m in MAIN)
        # The existing paper labels can be longer than internal variants;
        # replace metric cells only, not the entire labels.
        lines.append('% Experiment #{:d}: {}'.format(int(row['experiment_number']), row['experiment_id']))
        lines.append('% Dice & IoU & Precision & Recall & MAE & Loss: ' + values)
    (base / f'BT_UNET_PAPER_VALUES_{suffix}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return complete


def print_paper_result(record):
    values = ' | '.join(f'{k.upper()}={record[k]}' for k in MAIN + EXTRA)
    print(f'[PAPER RESULT #{record["experiment_number"]:03d}/165] '
          f'{record["dataset"]} {record["model"]} seed={record["seed"]} {values}', flush=True)
    print('[LATEX SIX CELLS] ' + ' & '.join('\\(' + record[k] + '\\)' for k in MAIN), flush=True)
