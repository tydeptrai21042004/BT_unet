#!/usr/bin/env python3
"""Evaluate the same BT checkpoint with no intervention and three band suppressions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import ROOT, write_csv
from src.analysis.frequency_interventions import BANDS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-root',required=True)
    p.add_argument('--config')
    p.add_argument('--device',default='auto')
    p.add_argument('--num-workers',type=int,default=2)
    args=p.parse_args()
    results=[]
    for label in ['none',*BANDS]:
        target=Path(args.output_root)/label
        cmd=[sys.executable,str(ROOT/'scripts/revision/evaluate_revision.py'),
             '--checkpoint',args.checkpoint,'--dataset',args.dataset,
             '--data-root',args.data_root,'--device',args.device,
             '--num-workers',str(args.num_workers),'--output-dir',str(target)]
        if args.config:cmd+=['--config',args.config]
        if label!='none':cmd+=['--remove-frequency-band',label]
        subprocess.run(cmd,check=True,cwd=ROOT)
        summary=json.loads((target/'summary.json').read_text())
        results.append({'removed_frequency_band':label,**summary['metrics_mean']})
    clean=results[0]['dice']
    for row in results:row['dice_drop_from_original']=clean-row['dice']
    write_csv(results,Path(args.output_root)/'frequency_interventions.csv')
