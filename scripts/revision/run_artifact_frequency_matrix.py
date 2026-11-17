#!/usr/bin/env python3
"""Frozen-checkpoint corruption × frequency-band intervention experiment.

Each cell performs evaluation on the same patient/image IDs; clean controls are
also run. Results are sensitivity measurements, not proof of artifact removal.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .common import ROOT,write_csv,write_json,sha256_file
except ImportError:
    from common import ROOT,write_csv,write_json,sha256_file
from src.analysis.corruptions import SUPPORTED
from src.analysis.frequency_interventions import BANDS


def build_matrix(corruptions,severities,bands):
    if 'clean' not in corruptions:
        corruptions=['clean']+corruptions
    if 'none' not in bands:
        bands=['none']+bands
    if any(c not in SUPPORTED for c in corruptions):
        raise ValueError(f'Choose corruptions from {SUPPORTED}')
    if any(b not in (*BANDS,'none') for b in bands):
        raise ValueError(f'Choose bands from {list(BANDS)+["none"]}')
    if any(int(x)<1 or int(x)>5 for x in severities):
        raise ValueError('Corruption severities must be 1 through 5')
    return [(c,1 if c=='clean' else int(s),b) for c in corruptions
            for s in ([1] if c=='clean' else severities) for b in bands]


def run(args):
    out=Path(args.output_root)
    corruptions=[s.strip() for s in args.corruptions.split(',') if s.strip()]
    severities=[int(s.strip()) for s in args.severities.split(',') if s.strip()]
    bands=[s.strip() for s in args.bands.split(',') if s.strip()]
    matrix=build_matrix(corruptions,severities,bands)
    outputs=[]
    for corruption,severity,band in matrix:
        folder=out/corruption/f'severity_{severity}'/band
        summary=folder/'summary.json'
        cmd=[sys.executable,str(ROOT/'scripts/revision/evaluate_revision.py'),
             '--checkpoint',args.checkpoint,'--dataset',args.dataset,
             '--data-root',args.data_root,'--split',args.split,
             '--corruption',corruption,'--severity',str(severity),
             '--seed',str(args.seed),'--device',args.device,
             '--num-workers',str(args.num_workers), '--output-dir',str(folder)]
        if args.config:cmd+=['--config',args.config]
        if band!='none':cmd+=['--remove-frequency-band',band]
        if args.run:
            subprocess.run(cmd,check=True,cwd=ROOT)
            if not summary.is_file():raise FileNotFoundError(summary)
            record=json.loads(summary.read_text())
            outputs.append({'corruption':corruption,'severity':severity,
                            'frequency_band_removed':band,
                            'checkpoint_sha256':record['checkpoint_sha256'],
                            'n_samples':record['num_samples'],
                            **record['metrics_mean']})
        else:
            print('PLAN',' '.join(cmd),flush=True)
    write_json({'status':'executed' if args.run else 'plan_only',
                'dataset':args.dataset,'split':args.split,
                'checkpoint_sha256':sha256_file(args.checkpoint),
                'n_jobs':len(matrix),'matrix':[list(item) for item in matrix],
                'note':'Corruption interventions on frozen model; not clinical causality.'},
               out/'artifact_frequency_manifest.json')
    if outputs:
        refs={(r['corruption'],r['severity']):r['dice'] for r in outputs if r['frequency_band_removed']=='none'}
        original=refs.get(('clean',1))
        for row in outputs:
            clean_control=refs.get((row['corruption'],row['severity']))
            row['dice_drop_due_to_band']=clean_control-row['dice'] if clean_control is not None else None
            row['dice_drop_due_to_corruption']=original-clean_control if original is not None and clean_control is not None else None
        write_csv(outputs,out/'artifact_frequency_results.csv')
    return len(outputs)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--config')
    p.add_argument('--dataset',required=True);p.add_argument('--data-root',default='data')
    p.add_argument('--split',default='test')
    p.add_argument('--corruptions',default='glare,motion_blur,low_contrast')
    p.add_argument('--severities',default='1,3')
    p.add_argument('--bands',default='none,low,middle,high')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--device',default='auto');p.add_argument('--output-root',required=True)
    p.add_argument('--run',action='store_true',help='Default: only plan; add to execute')
    a=p.parse_args();print('Completed evaluations:',run(a))
