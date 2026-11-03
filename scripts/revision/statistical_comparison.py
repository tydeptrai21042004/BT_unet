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
    """Paired differences with seed-level tests and optional cluster uncertainty.

    Repeated images from the same training seed are NOT independent training
    replicates. For inference over independent training runs, aggregate per seed
    and test those differences. A 3-pair exact Wilcoxon test cannot get p<.05.
    """
    a, b = read_rows(a), read_rows(b)
    if set(a) != set(b):
        missing_a = list(set(b)-set(a))[:5]
        missing_b = list(set(a)-set(b))[:5]
        raise ValueError(f'Unmatched pairs: candidate missing={missing_a}; reference missing={missing_b}')
    if resamples < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    keys = sorted(a)
    diff = np.array([float(a[k][metric])-float(b[k][metric]) for k in keys])
    if not np.isfinite(diff).all():
        raise ValueError('Nonfinite paired differences')
    seed_values = sorted(set(k[1] for k in keys))
    by_seed = np.array([diff[np.array([key[1] == s for key in keys])].mean() for s in seed_values])
    n_seeds = len(seed_values)
    rng = np.random.default_rng(seed)
    if n_seeds >= 2:
        boot_seed = rng.choice(by_seed, size=(resamples,n_seeds), replace=True).mean(axis=1)
        seed_ci = [float(np.quantile(boot_seed, q)) for q in (.025,.975)]
    else:
        seed_ci = [float('nan')]*2
    run_t_p = float(stats.ttest_1samp(by_seed,0.0).pvalue) if n_seeds >= 2 and float(by_seed.std()) > 1e-12 else float('nan')
    try:
        run_wilcoxon_p = float(stats.wilcoxon(by_seed,zero_method='wilcox',alternative='two-sided',method='exact').pvalue) if n_seeds >= 2 and np.any(by_seed != 0) else float('nan')
    except (ValueError,TypeError):
        run_wilcoxon_p = float('nan')
    # Cluster bootstrap is conditional on selected patients/videos, independent
    # of run-level CI. Without a known cluster ID it is image-level exploratory.
    groups = [a[k].get(cluster_field, '') if cluster_field else k[0] for k in keys]
    if cluster_field and any(not g for g in groups):
        raise ValueError(f'Missing cluster ID: {cluster_field}')
    if cluster_field and any(a[k].get(cluster_field)!=b[k].get(cluster_field) for k in keys):
        raise ValueError('Candidate and reference patient/video cluster IDs disagree')
    unique = sorted(set(groups))
    values = np.array(groups)
    by_group = {g: diff[values == g] for g in unique}
    boot=[]
    for _ in range(resamples):
        drawn=rng.choice(unique,size=len(unique),replace=True)
        boot.append(float(np.concatenate([by_group[g] for g in drawn]).mean()))
    return {'n_pairs':len(diff),'n_clusters':len(unique),'n_training_seeds':n_seeds,
            'mean_difference':float(diff.mean()),'median_difference':float(np.median(diff)),
            'bootstrap_ci_low':float(np.quantile(boot,.025)),
            'bootstrap_ci_high':float(np.quantile(boot,.975)),
            'training_seed_ci_low':seed_ci[0], 'training_seed_ci_high':seed_ci[1],
            'paired_t_p_exploratory':run_t_p,
            'paired_wilcoxon_p_training_seeds':run_wilcoxon_p,
            'note':('Paired tests use independent seed-mean differences, NOT per-image pseudoreplicates. '
                    'Bootstrap confidence limits resample provided patient/video clusters '
                    'when cluster_field is supplied; otherwise image IDs. '
                    'With only 3 seeds exact two-sided Wilcoxon cannot reach p<0.05.')}


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
