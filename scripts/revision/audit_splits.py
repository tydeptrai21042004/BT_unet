#!/usr/bin/env python3
"""Fail-closed manifest audit: no silent train/test overlap or split fallback."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    from .common import sha256_file, write_json
except ImportError:
    from common import sha256_file, write_json
from src.datasets import build_dataset, normalize_dataset_name


def manifest_path(root, dataset, split, strict_dataset_paths=False):
    root=Path(root)
    candidates=[root/'splits'/dataset/f'{split}.txt',root/'splits'/f'{dataset}_{split}.txt']
    if not strict_dataset_paths:
        candidates.append(root/'splits'/f'{split}.txt')
    for item in candidates:
        if item.exists():return item
    raise FileNotFoundError(f'Missing {dataset}/{split} manifest: searched {candidates}. Explicit manifests required in revision protocol.')


def audit(root, datasets, image_size, compare_content=False, require_all=True):
    report={'datasets':{},'overlap_ids':[],'overlap_files':[]}
    seen={}  # content digest -> (dataset, split, sample id)
    for dataset in datasets:
        names=['train','val','test'] if require_all else ['test']
        sets={}
        for split in names:
            path=manifest_path(root,dataset,split)
            rows=[s.strip().lower() for s in path.read_text().splitlines() if s.strip() and not s.strip().startswith('#')]
            if len(rows)!=len(set(rows)):
                raise ValueError(f'Duplicate IDs in {path}')
            if not rows:raise ValueError(f'Empty split {path}')
            sets[split]=set(rows)
            report['datasets'].setdefault(dataset,{})[split]={'count':len(rows),'sha256':sha256_file(path)}
            if compare_content:
                data=build_dataset(dataset,root,split=split,image_size=image_size)
                for sid,img,_ in data.samples:
                    digest=sha256_file(img)
                    if digest in seen and seen[digest] != (dataset,split,sid):
                        report['overlap_files'].append({'first':seen[digest],'second':(dataset,split,sid),'sha256':digest})
                    seen[digest]=(dataset,split,sid)
        for a in names:
            for b in names:
                if a>=b:continue
                same=sets[a]&sets[b]
                if same:report['overlap_ids'].append({'dataset':dataset,'splits':[a,b],'ids':sorted(same)[:12]})
    if report['overlap_ids'] or report['overlap_files']:
        raise ValueError(f'Data leakage: {report["overlap_ids"][:3]} {report["overlap_files"][:3]}')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data')
    p.add_argument('--datasets',required=True,help='Comma separated dataset names')
    p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--compare-content-hashes',action='store_true')
    p.add_argument('--test-only',action='store_true',help='For external-only test sets')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    datasets=[normalize_dataset_name(v.strip()) for v in a.datasets.split(',') if v.strip()]
    report=audit(a.data_root,datasets,a.image_size,a.compare_content_hashes,not a.test_only)
    write_json(report,a.output);print('PASS:',a.output)
