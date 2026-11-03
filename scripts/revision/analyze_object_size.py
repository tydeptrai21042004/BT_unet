#!/usr/bin/env python3
"""Per-size segmentation results from real per-image evaluation CSVs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    from .common import write_csv,write_json
except ImportError:
    from common import write_csv,write_json


def analyze(paths, lower=.01, upper=.10):
    if not 0 < lower < upper < 1:
        raise ValueError('Size thresholds must satisfy 0 < lower < upper < 1')
    data=defaultdict(list)
    for pair in paths:
        if '=' not in pair:raise ValueError('Use NAME=/path/per_image.csv')
        name,path=pair.split('=',1)
        rows=list(csv.DictReader(open(path,newline='',encoding='utf-8')))
        if not rows:raise ValueError(f'Empty results in {path}')
        for r in rows:
            frac=float(r['lesion_fraction'])
            size='empty' if frac==0 else ('small' if frac<lower else ('medium' if frac<upper else 'large'))
            data[(name,str(r['seed']),size)].append(r)
    out=[]
    for (name,seed,size),values in sorted(data.items()):
        item={'model':name,'seed':seed,'size_category':size,'n_images':len(values)}
        for metric in ('dice','iou','boundary_f1','recall','complete_miss'):
            scores=[float(r[metric]) for r in values if metric in r and np.isfinite(float(r[metric]))]
            item[metric]=float(np.mean(scores)) if scores else float('nan')
        out.append(item)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,help='MODEL=per_image.csv')
    p.add_argument('--small-upper',type=float,default=.01)
    p.add_argument('--medium-upper',type=float,default=.10)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=analyze(a.runs,a.small_upper,a.medium_upper)
    write_csv(result,a.output)
    write_json({'size_thresholds':{'small_upper_exclusive':a.small_upper,
                                   'medium_upper_exclusive':a.medium_upper},
                'note':'Strata assigned from ground truth lesion_fraction at evaluation resolution; no posthoc threshold selection.'},
               Path(a.output).with_suffix('.json'))
