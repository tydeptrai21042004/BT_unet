#!/usr/bin/env python3
"""Nested, deterministic fractions of an existing training manifest; val/test remain untouched."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import write_json, sha256_file
from audit_splits import manifest_path
from src.datasets import normalize_dataset_name


def make_fractions(root, dataset, fractions, seed, output):
    source=manifest_path(root,dataset,'train')
    manifest_path(root,dataset,'val');manifest_path(root,dataset,'test')
    ids=[v.strip() for v in source.read_text().splitlines() if v.strip() and not v.startswith('#')]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate IDs in training manifest')
    rng=random.Random(seed);ids=sorted(ids);rng.shuffle(ids)
    files={}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for f in fractions:
        if not 0 < f <= 1:raise ValueError(f'fraction out of range: {f}')
        n=max(1,round(f*len(ids)))
        dest=out/f'train_{int(round(100*f)):03d}pct_seed{seed}.txt'
        dest.write_text('\n'.join(sorted(ids[:n]))+'\n',encoding='utf-8')
        files[str(f)]={'path':str(dest),'num_train':n,'sha256':sha256_file(dest)}
    write_json({'source_manifest':str(source),'source_sha256':sha256_file(source),
                'seed':seed,'fractions':files},out/'manifest.json')
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data')
    p.add_argument('--dataset',required=True)
    p.add_argument('--fractions',default='0.1,0.25,0.5,1.0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    print(make_fractions(a.data_root,normalize_dataset_name(a.dataset),
                         [float(x) for x in a.fractions.split(',')],a.seed,a.output_dir))
