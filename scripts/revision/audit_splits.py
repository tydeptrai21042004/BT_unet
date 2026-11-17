#!/usr/bin/env python3
"""Strict split audit with sample, file-content and patient/video leakage checks.

No generic split fallback is allowed for reviewer experiments. Optional CSV maps
must cover all samples and contain ``sample_id,cluster_id``. A cluster cannot
appear in two splits of the same dataset. No patient inference from filenames.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .common import sha256_file, write_json
except ImportError:
    from common import sha256_file, write_json
from src.datasets import build_dataset, normalize_dataset_name


def manifest_path(root, dataset, split, strict_dataset_paths=True):
    root = Path(root)
    candidates = [root/'splits'/dataset/f'{split}.txt', root/'splits'/f'{dataset}_{split}.txt']
    if not strict_dataset_paths:
        candidates.append(root/'splits'/f'{split}.txt')
    for item in candidates:
        if item.is_file():
            return item
    raise FileNotFoundError(f'Missing {dataset}/{split} manifest: {candidates}')


def read_cluster_map(path):
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not {'sample_id','cluster_id'}.issubset(rows[0]):
        raise ValueError(f'{path}: require sample_id,cluster_id CSV header and rows')
    result = {}
    for row in rows:
        sid = row['sample_id'].strip().lower()
        group = row['cluster_id'].strip()
        if not sid or not group or sid in result:
            raise ValueError(f'Invalid or duplicated cluster mapping {sid!r} in {path}')
        result[sid] = group
    return result


def audit(root, datasets, image_size=352, compare_content=False, require_all=True,
          cluster_maps=None):
    """Return audit on success; fail closed on detected overlap.

    ``cluster_maps`` is optional dataset -> CSV path. Without a real mapping,
    no patient/video-disjointness is claimed.
    """
    root = Path(root)
    cluster_maps = cluster_maps or {}
    report = {'datasets':{}, 'overlap_ids':[], 'overlap_files':[],
              'overlap_clusters':[], 'patient_video_integrity_verified':{}}
    seen_files = {} # content hash -> first (dataset, split, id)
    for dataset in datasets:
        names = ['train','val','test'] if require_all else ['test']
        sets = {}
        map_path = cluster_maps.get(dataset)
        if map_path is None:
            candidate = root/'splits'/dataset/'clusters.csv'
            if candidate.is_file():
                map_path = candidate
        cluster_map = read_cluster_map(map_path) if map_path else None
        report['patient_video_integrity_verified'][dataset] = bool(cluster_map)
        groups_by_split = {}
        for split in names:
            path = manifest_path(root,dataset,split,strict_dataset_paths=True)
            lines = [s.strip().lower() for s in path.read_text(encoding='utf-8').splitlines()
                     if s.strip() and not s.lstrip().startswith('#')]
            if not lines or len(set(lines)) != len(lines):
                raise ValueError(f'Empty or duplicate sample IDs in {path}')
            sets[split] = set(lines)
            report['datasets'].setdefault(dataset,{})[split] = {
                'count':len(lines), 'manifest':str(path), 'sha256':sha256_file(path)}
            if cluster_map is not None:
                missing = sets[split]-cluster_map.keys()
                if missing:
                    raise ValueError(f'Unknown clusters for {dataset}/{split}: {sorted(missing)[:10]}')
                groups_by_split[split] = {cluster_map[s] for s in sets[split]}
            if compare_content:
                # Build the actual underlying file paths so differing IDs with
                # byte-identical content are detected, including cross-dataset.
                data = build_dataset(dataset,root,split=split,image_size=image_size)
                found = set()
                for sid, image, _mask in data.samples:
                    sid = str(sid).lower()
                    if sid not in sets[split]:
                        raise ValueError(f'Actual dataset sample {sid} not in manifest {path}')
                    found.add(sid)
                    digest = sha256_file(image)
                    prior = seen_files.setdefault(digest,(dataset,split,sid))
                    if prior != (dataset,split,sid):
                        report['overlap_files'].append({'first':prior,'second':(dataset,split,sid),'sha256':digest})
                if found != sets[split]:
                    raise ValueError(f'Manifest and materialized samples differ for {dataset}/{split}')
        for i,left in enumerate(names):
            for right in names[i+1:]:
                overlap = sets[left] & sets[right]
                if overlap:
                    report['overlap_ids'].append({'dataset':dataset,'splits':[left,right],
                                                   'ids':sorted(overlap)[:20]})
                if cluster_map is not None:
                    overlap_groups = groups_by_split[left] & groups_by_split[right]
                    if overlap_groups:
                        report['overlap_clusters'].append({'dataset':dataset,'splits':[left,right],
                                                           'clusters':sorted(overlap_groups)[:20]})
        if cluster_map is not None:
            unused = cluster_map.keys()-set.union(*sets.values())
            report['datasets'][dataset]['cluster_map'] = {'path':str(map_path),
                'sha256':sha256_file(map_path),'unused_count':len(unused)}
    if report['overlap_ids'] or report['overlap_files'] or report['overlap_clusters']:
        raise ValueError('Data leakage detected: '+str({key:report[key][:3] for key in
                 ('overlap_ids','overlap_files','overlap_clusters')}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data')
    p.add_argument('--datasets',required=True,help='Comma-separated dataset names')
    p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--compare-content-hashes',action='store_true')
    p.add_argument('--test-only',action='store_true')
    p.add_argument('--cluster-map',action='append',default=[],metavar='DATASET=FILE',
                   help='Patient/video cluster mappings; may be repeated')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    datasets = [normalize_dataset_name(s.strip()) for s in a.datasets.split(',') if s.strip()]
    maps = {}
    for mapping in a.cluster_map:
        if '=' not in mapping:
            p.error('--cluster-map requires DATASET=FILE')
        key, value = mapping.split('=',1)
        maps[normalize_dataset_name(key)] = value
    result = audit(a.data_root,datasets,a.image_size,a.compare_content_hashes,
                   not a.test_only,maps)
    write_json(result,a.output)
    print('PASS:',a.output)
