#!/usr/bin/env python3
"""Aggregate actual training history over repeated seeds, without smoothing."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    from .common import write_csv
except ImportError:
    from common import write_csv


def aggregate_histories(items: list[str], metric='val/dice') -> list[dict]:
    grouped=defaultdict(list)
    for item in items:
        if '=' not in item:
            raise ValueError(f'Use MODEL=/path/history.csv: {item}')
        label, file = item.split('=', 1)
        rows=list(csv.DictReader(open(file,newline='',encoding='utf-8')))
        if not rows or metric not in rows[0]:
            raise ValueError(f'Empty/missing metric {metric!r} in {file}')
        curve={int(float(r.get('epoch') or i+1)):float(r[metric]) for i,r in enumerate(rows)}
        if len(curve) != len(rows):
            raise ValueError(f'Duplicate epochs in {file}')
        grouped[label].append(curve)
    output=[]
    for label,curves in sorted(grouped.items()):
        shared=sorted(set.intersection(*(set(c) for c in curves)))
        if not shared:
            raise ValueError(f'No overlapping epochs among runs of {label}')
        for epoch in shared:
            vals=np.array([c[epoch] for c in curves],float)
            if not np.isfinite(vals).all():
                raise ValueError(f'Nonfinite {metric} in {label} epoch {epoch}')
            output.append({'model':label,'epoch':epoch,'n_seeds':len(vals),
                           'mean':float(vals.mean()),
                           'std':float(vals.std(ddof=1)) if len(vals)>1 else 0.0})
    return output


def plot(rows, output, metric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=sorted({r['model'] for r in rows})
    fig, ax=plt.subplots(figsize=(7.4,4.5))
    for label in labels:
        subset=[r for r in rows if r['model']==label]
        x=np.array([r['epoch'] for r in subset])
        mu=np.array([r['mean'] for r in subset])
        sd=np.array([r['std'] for r in subset])
        (line,) = ax.plot(x,mu,label=f'{label} (n={subset[0]["n_seeds"]})')
        if subset[0]['n_seeds']>1:
            ax.fill_between(x,mu-sd,mu+sd,color=line.get_color(),alpha=.16)
    ax.set(xlabel='Epoch',ylabel=metric)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=220);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='Repeated MODEL=/path/history.csv')
    p.add_argument('--metric',default='val/dice')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rows=aggregate_histories(args.runs,args.metric)
    plot(rows,args.output,args.metric)
    write_csv(rows,Path(args.output).with_suffix('.csv'))
