#!/usr/bin/env python3
"""Source-only training on multiple datasets, followed by held-out target evaluation.

Requires prepared images and explicit split manifests. Never generates test results
by training on a target split. Optional dry-run prints commands before execution.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, write_json, sha256_file
from audit_splits import audit, manifest_path
from src.datasets import normalize_dataset_name, build_dataset


def comma(value):
    return [normalize_dataset_name(x.strip()) for x in value.split(',') if x.strip()]


def plan(args):
    sources,targets=comma(args.sources),comma(args.targets)
    if not sources or not targets or len(set(sources))!=len(sources) or len(set(targets))!=len(targets):
        raise ValueError('Need nonempty unique source and target dataset lists')
    if set(sources)&set(targets):
        raise ValueError('External target dataset must be disjoint from all training sources')
    if args.val_source not in sources:raise ValueError('Validation source must be one of the training sources')
    for name in sources:
        for split in ('train','val','test'):manifest_path(args.data_root,name,split)
    for name in targets:manifest_path(args.data_root,name,'test')
    # Detect within-dataset overlaps and, optionally, exact duplicate images.
    audit(args.data_root,sources,args.image_size,compare_content=args.compare_content_hashes,require_all=True)
    audit(args.data_root,targets,args.image_size,compare_content=args.compare_content_hashes,require_all=False)
    if args.compare_content_hashes:
        # Explicitly test *cross-domain* overlap, not just per-dataset partitions.
        source_hashes = {}
        for name in sources:
            for split in ('train', 'val', 'test'):
                data = build_dataset(name, args.data_root, split=split, image_size=args.image_size)
                for sid, img, _ in data.samples:
                    source_hashes.setdefault(sha256_file(img), (name, split, sid))
        for name in targets:
            data = build_dataset(name, args.data_root, split='test', image_size=args.image_size)
            for sid, img, _ in data.samples:
                digest = sha256_file(img)
                if digest in source_hashes:
                    raise ValueError(f'Exact source/target image overlap: {source_hashes[digest]} vs {(name, "test", sid)}')
    models=[v.strip() for v in args.models.split(',') if v.strip()]
    seeds=[int(v.strip()) for v in args.seeds.split(',') if v.strip()]
    runs=[]
    for model in models:
        config=Path(args.config_dir)/f'{model}.yaml'
        if not config.exists():raise FileNotFoundError(config)
        for seed in seeds:
            run_root=Path(args.output_root)/model/f'seed_{seed}'
            train=[sys.executable,str(ROOT/'scripts/train_one.py'), '--model',model, '--config',str(config),
                   '--dataset',sources[0], '--train-datasets',','.join(sources), '--val-dataset',args.val_source,
                   '--data-root',str(args.data_root),'--output-root',str(run_root),'--seed',str(seed),
                   '--epochs',str(args.epochs),'--image-size',str(args.image_size),
                   '--device',args.device, '--num-workers',str(args.num_workers)]
            ckpt=run_root/model/'checkpoints'/'best.pt'
            evaluations=[]
            for target in targets:
                eval_root=run_root/'external_tests'/target
                command=[sys.executable,str(ROOT/'scripts/revision/evaluate_revision.py'),
                         '--checkpoint',str(ckpt),'--dataset',target,'--data-root',str(args.data_root),
                         '--split','test','--image-size',str(args.image_size),'--device',args.device,
                         '--num-workers',str(args.num_workers),'--seed',str(seed),
                         '--output-dir',str(eval_root),'--save-predictions']
                evaluations.append(command)
            runs.append({'model':model,'seed':seed,'training':train,'evaluations':evaluations,'checkpoint':str(ckpt)})
    return runs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',default='kvasir_seg,cvc_clinicdb')
    p.add_argument('--targets',default='cvc_colondb,etis,cvc_300')
    p.add_argument('--val-source',default='kvasir_seg')
    p.add_argument('--models',default='unet,unetpp,resunetpp,dapr_unet')
    p.add_argument('--config-dir',default='configs/fair')
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-root',default='outputs_revision/cross_dataset')
    p.add_argument('--seeds',default='42,1,2')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--device',default='auto')
    p.add_argument('--compare-content-hashes',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    args.val_source=normalize_dataset_name(args.val_source)
    runs=plan(args)
    root=Path(args.output_root)
    write_json({'sources':comma(args.sources),'targets':comma(args.targets),
                'validation_source':args.val_source,'runs':runs},root/'plan.json')
    for job in runs:
        commands=[job['training'],*job['evaluations']]
        for command in commands:
            print('[DRY RUN]' if args.dry_run else '[RUN]',subprocess.list2cmdline(command),flush=True)
            if not args.dry_run:
                subprocess.run(command,check=True,cwd=ROOT)
