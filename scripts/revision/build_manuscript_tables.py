#!/usr/bin/env python3
"""Manuscript tables strictly from completed reviewer run outputs.

Tables use image-average metrics per independent training run, then report
mean/std across seeds; no results are inferred for missing checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    from .common import write_csv, write_json
except ImportError:
    from common import write_csv, write_json

METRICS=['dice','iou','precision','recall','mae','boundary_f1','hd95_pixels','assd_pixels',
         'complete_miss','empty_prediction']


def summarize(plan_path: Path, *, strict=True):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    groups=defaultdict(list)
    missing=[]
    for job in plan['jobs']:
        csv_path=Path(job['metrics_dir'])/'per_image.csv'
        if not csv_path.is_file():
            missing.append(str(csv_path));continue
        with csv_path.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        if not rows:
            raise ValueError(f'Empty per-image file: {csv_path}')
        if len(rows)!=len({r['sample_id'] for r in rows}):
            raise ValueError(f'Duplicate sample IDs within evaluation {csv_path}')
        if any(int(r['seed'])!=int(job['seed']) for r in rows):
            raise ValueError(f'Checkpoint training seed mismatches run plan: {csv_path}')
        summary={'dataset':job['dataset'],'group':job['group'],
                 'variant':job['variant'],'model':job['model'],'seed':job['seed'],
                 'n_images':len(rows)}
        for key in METRICS:
            if key not in rows[0]:
                continue
            nums=np.array([float(row[key]) for row in rows],float)
            valid=nums[np.isfinite(nums)]
            summary[key]=float(valid.mean()) if valid.size else float('nan')
            summary[f'{key}_n_finite']=int(valid.size)
        groups[(job['dataset'],job['group'],job['variant'])].append(summary)
    if strict and missing:
        raise FileNotFoundError(f'{len(missing)} incomplete evaluations; first missing: {missing[:3]}')
    per_seed=[r for entries in groups.values() for r in entries]
    aggregated=[]
    for (dataset,group,variant),items in sorted(groups.items()):
        if len({r['seed'] for r in items})!=len(items):
            raise ValueError(f'Duplicate seeds for {dataset}/{group}/{variant}')
        row={'dataset':dataset,'group':group,'variant':variant,
             'n_seeds':len(items),'seeds':','.join(str(i) for i in sorted(r['seed'] for r in items)),
             'n_images_per_seed':','.join(str(r['n_images']) for r in items)}
        for key in METRICS:
            values=np.array([r.get(key,float('nan')) for r in items],float)
            finite=values[np.isfinite(values)]
            row[f'{key}_mean']=float(finite.mean()) if len(finite) else float('nan')
            row[f'{key}_std']=float(finite.std(ddof=1)) if len(finite)>1 else float('nan')
        aggregated.append(row)
    return per_seed,aggregated,missing


def tex_table(rows, metric='dice'):
    esc=lambda s:str(s).replace('_',r'\_').replace('&',r'\&')
    parts=[r'\begin{tabular}{lllc}',r'\toprule',
           r'Dataset & Group & Variant & '+esc(metric)+r' $\uparrow$ \\',r'\midrule']
    for r in rows:
        mean=r.get(f'{metric}_mean',float('nan'))
        std=r.get(f'{metric}_std',float('nan'))
        val=f'{mean:.4f} $\\pm$ {std:.4f}' if np.isfinite(std) else (f'{mean:.4f} (n={r["n_seeds"]})' if np.isfinite(mean) else 'N/A')
        parts.append(f'{esc(r["dataset"])} & {esc(r["group"])} & {esc(r["variant"])} & {val} '+r'\\')
    parts += [r'\bottomrule',r'\end{tabular}']
    return '\n'.join(parts)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--metric',default='dice',choices=METRICS)
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    per_seed,aggregate,missing=summarize(Path(args.plan),strict=not args.allow_incomplete)
    if not per_seed:
        raise ValueError('No completed runs to aggregate')
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    write_csv(per_seed,root/'per_seed_metrics.csv')
    write_csv(aggregate,root/'multi_seed_metrics.csv')
    (root/'manuscript_table.tex').write_text(tex_table(aggregate,args.metric),encoding='utf-8')
    write_json({'complete_runs':len(per_seed),'missing_count':len(missing),'missing':missing},root/'aggregation_status.json')
    print('Wrote measured multi-seed tables to',root)
