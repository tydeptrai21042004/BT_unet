#!/usr/bin/env python3
"""Pre-declared deterministic selection of difficult and typical test images.

Never chooses examples based solely on BT beating another method. Requires
matching sample IDs in all supplied evaluations.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .common import write_json
except ImportError:
    from common import write_json


def choose(path, limit=8):
    rows=list(csv.DictReader(open(path,newline='',encoding='utf-8')))
    if not rows:raise ValueError('Empty case CSV')
    for key in ['sample_id','lesion_fraction','dice','boundary_f1','precision','recall']:
        if key not in rows[0]:raise KeyError(key)
    if len(set(r['sample_id'] for r in rows)) != len(rows):
        raise ValueError('Expected one evaluation seed and unique sample IDs')
    rows=sorted(rows,key=lambda r:r['sample_id'])
    ranked=[('small_lesion',sorted(rows,key=lambda r:(float(r['lesion_fraction']),r['sample_id']))),
            ('large_lesion',sorted(rows,key=lambda r:(-float(r['lesion_fraction']),r['sample_id']))),
            ('difficult_boundary',sorted(rows,key=lambda r:(float(r['boundary_f1']),r['sample_id']))),
            ('low_dice',sorted(rows,key=lambda r:(float(r['dice']),r['sample_id']))),
            ('low_precision',sorted(rows,key=lambda r:(float(r['precision']),r['sample_id']))),
            ('low_recall',sorted(rows,key=lambda r:(float(r['recall']),r['sample_id']))),
            ('typical',sorted(rows,key=lambda r:(abs(float(r['dice'])-.8),r['sample_id'])))]
    chosen=[];used=set()
    for label, options in ranked:
        if len(chosen)>=limit:break
        for row in options:
            if row['sample_id'] not in used:
                chosen.append({'sample_id':row['sample_id'],'selection_reason':label,
                               'dice_reference':float(row['dice'])})
                used.add(row['sample_id']);break
    for row in rows:
        if len(chosen)>=limit:break
        if row['sample_id'] not in used:
            chosen.append({'sample_id':row['sample_id'],'selection_reason':'remaining_alphabetical',
                           'dice_reference':float(row['dice'])})
            used.add(row['sample_id'])
    return chosen

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,help='Reference U-Net per_image.csv (single seed)')
    p.add_argument('--max-cases',type=int,default=8)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    cases=choose(args.reference,args.max_cases)
    write_json({'selection_basis':'Reference model difficulty/size (not BT relative advantage)',
                'cases':cases,'ids':','.join(x['sample_id'] for x in cases)},args.output)
    print('IDs for plot_qualitative.py --ids:',','.join(x['sample_id'] for x in cases))
