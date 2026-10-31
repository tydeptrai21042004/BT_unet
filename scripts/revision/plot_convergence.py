#!/usr/bin/env python3
"""Plot genuinely recorded history.csv data; no smoothing or synthetic interpolation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(8,5))
    metadata=[]
    for item in args.runs:
        label,src=item.split('=',1)
        with open(src,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if not rows:raise ValueError(f'Empty history: {src}')
        if args.metric not in rows[0]:raise KeyError(f'{args.metric} not in {src}; keys={list(rows[0])}')
        x=[int(float(r.get('epoch',i+1))) for i,r in enumerate(rows)]
        y=[float(r[args.metric]) for r in rows]
        ax.plot(x,y,label=label)
        metadata.append((label,len(rows),max(y) if 'dice' in args.metric else min(y)))
    ax.set(xlabel='Epoch',ylabel=args.metric,title='Recorded validation convergence')
    ax.legend();ax.grid(alpha=.2)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)
    print('Saved',path,'Runs:',metadata)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='NAME=/path/to/history.csv')
    p.add_argument('--metric',default='val/dice')
    p.add_argument('--output',required=True)
    main(p.parse_args())
