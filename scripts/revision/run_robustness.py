#!/usr/bin/env python3
"""Run deterministic corruptions on one frozen checkpoint; export mean Dice drops."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from common import ROOT, write_csv
from src.analysis.corruptions import SUPPORTED


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config')
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-root',required=True)
    p.add_argument('--corruptions',default=','.join(SUPPORTED))
    p.add_argument('--severities',default='1,2,3,4')
    p.add_argument('--device',default='auto')
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    output=Path(args.output_root)
    results=[]
    for corruption in args.corruptions.split(','):
        if corruption not in SUPPORTED:raise ValueError(corruption)
        for severity in ([1] if corruption=='clean' else [int(v) for v in args.severities.split(',')]):
            sub=output/corruption/f'severity_{severity}'
            cmd=[sys.executable,str(ROOT/'scripts/revision/evaluate_revision.py'),
                 '--checkpoint',args.checkpoint,'--dataset',args.dataset,'--data-root',args.data_root,
                 '--corruption',corruption,'--severity',str(severity),
                 '--device',args.device,'--num-workers',str(args.num_workers),
                 '--seed',str(args.seed),'--output-dir',str(sub)]
            if args.config:cmd+=['--config',args.config]
            subprocess.run(cmd,check=True,cwd=ROOT)
            import json
            summary=json.loads((sub/'summary.json').read_text())
            results.append({'corruption':corruption,'severity':severity,**summary['metrics_mean']})
    clean=next((r['dice'] for r in results if r['corruption']=='clean'),None)
    for row in results:
        row['dice_drop']=clean-row['dice'] if clean is not None else float('nan')
    write_csv(results,output/'robustness_summary.csv')
    print('Saved',output/'robustness_summary.csv')
