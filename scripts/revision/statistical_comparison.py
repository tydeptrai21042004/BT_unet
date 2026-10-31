#!/usr/bin/env python3
"""Paired comparisons on matched sample IDs, with uncertainty and Holm adjustment.

Use independent cluster IDs for videos/patients when available; otherwise image
bootstrap intervals are conditional on sampled images, not patient independence.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from common import write_csv, write_json


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'Empty CSV: {path}')
    if len(rows)!=len({(r['sample_id'],r.get('seed','')) for r in rows}):
        raise ValueError('Duplicate sample_id/seed pairs, check inputs')
    return {(r['sample_id'],r.get('seed','')):r for r in rows}


def compare(a, b, metric='dice', cluster_field=None, resamples=2000, seed=123):
    a,b=read_rows(a),read_rows(b)
    if set(a)!=set(b):
        missing_a=list(set(b)-set(a))[:5];missing_b=list(set(a)-set(b))[:5]
        raise ValueError(f'Unmatched samples: missing from first={missing_a}, from second={missing_b}')
    keys=sorted(a)
    diff=np.array([float(a[k][metric])-float(b[k][metric]) for k in keys])
    if not np.isfinite(diff).all():raise ValueError('Nonfinite metric difference')
    # Group repeated observations from same patient/video if requested.
    groups = [a[k].get(cluster_field, '') if cluster_field else f'{k[0]}:{k[1]}' for k in keys]
    if cluster_field and any(not value for value in groups):
        raise ValueError(f'Missing cluster ID {cluster_field} in per-image CSV')
    unique=sorted(set(groups));by_group={g:diff[np.array(groups)==g] for g in unique}
    rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(resamples):
        drawn=rng.choice(unique,size=len(unique),replace=True)
        boot.append(float(np.concatenate([by_group[g] for g in drawn]).mean()))
    pvalue=float(stats.ttest_1samp(diff,0).pvalue) if len(diff)>=3 else float('nan')
    return {'n_pairs':len(diff),'n_clusters':len(unique),'mean_difference':float(diff.mean()),
            'median_difference':float(np.median(diff)),
            'bootstrap_ci_low':float(np.quantile(boot,.025)),
            'bootstrap_ci_high':float(np.quantile(boot,.975)),
            'paired_t_p_exploratory':pvalue,
            'note':'Exploratory paired t test assumes independent paired differences; bootstrap resamples selected clusters.'}


def holm(p_values):
    out=[float('nan')]*len(p_values)
    valid=sorted((p,i) for i,p in enumerate(p_values) if np.isfinite(p))
    running=0.0
    for rank,(p,i) in enumerate(valid):
        running=max(running,min(1.0,p*(len(valid)-rank)))
        out[i]=running
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,help='Baseline per_image.csv')
    p.add_argument('--candidates',nargs='+',required=True,help='NAME=per_image.csv')
    p.add_argument('--metric',default='dice')
    p.add_argument('--cluster-field',help='Column with independent patient/video IDs')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rows=[]
    for item in args.candidates:
        name,path=item.split('=',1)
        result=compare(path,args.reference,args.metric,args.cluster_field,args.bootstrap,args.seed)
        rows.append({'candidate':name,'reference':args.reference,'metric':args.metric,**result})
    for row,adjusted in zip(rows,holm([r['paired_t_p_exploratory'] for r in rows])):
        row['holm_adjusted_p_exploratory']=adjusted
    write_csv(rows,args.output)
    print('Saved',args.output)
