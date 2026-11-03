#!/usr/bin/env python3
"""Reproducible plan/runner for reviewer experiments (no fabricated results).

DRY RUN is the default. Add --run to train/evaluate. Every model uses the
original train_one.py and the checkpoint-grounded revision evaluator.
Published external spectral models must pass strict source/audit gate first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.revision.common import sha256_file, write_json

FACTORIAL = [
    'dapr_unet', 'dapr_no_global_amplitude', 'dapr_no_global_phase',
    'dapr_no_global_channel_mix', 'dapr_amplitude_only', 'dapr_phase_only',
    'dapr_mixing_only', 'dapr_identity_spectral_control',
]
RESIDUAL = [
    ('direct', 'dapr_unet', None),
    ('additive_a025', 'dapr_matched_residual', 0.25),
    ('additive_a050', 'dapr_matched_residual', 0.50),
    ('additive_a100', 'dapr_matched_residual', 1.00),
    ('additive_zero_a050', 'dapr_matched_residual_zero_init', 0.50),
    ('difference_a050', 'dapr_matched_difference', 0.50),
    ('spatial_mixer', 'dapr_spatial_channel_mix_control', None),
]
CONVERGENCE = ['unet', 'unetpp', 'resunetpp', 'dapr_unet']


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Missing config: {path}')
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
        raise ValueError(f'Invalid experiment YAML: {path}')
    return payload


def models_for_group(group: str, external_configs: list[Path]):
    if group == 'factorial':
        for name in FACTORIAL:
            yield name, name, None, ROOT / 'configs/revision/dapr_unet.yaml'
    elif group == 'residual':
        for label, model, alpha in RESIDUAL:
            yield label, model, alpha, ROOT / 'configs/revision/dapr_unet.yaml' if model.startswith('dapr_spatial') or model == 'dapr_unet' else ROOT / 'configs/revision/dapr_matched_residual.yaml'
    elif group == 'convergence':
        for name in CONVERGENCE:
            yield name, name, None, ROOT / 'configs/fair' / f'{name}.yaml'
    elif group == 'official':
        if not external_configs:
            raise ValueError('--group official requires --external-configs (audited YAML, no fake baselines)')
        for path in external_configs:
            cfg = load_config(path)
            if cfg['model'].get('name') != 'verified_external_spectral':
                raise ValueError(f'Expected verified_external_spectral model in {path}')
            label = cfg.get('experiment', {}).get('name')
            if not label or label == 'verified_external_spectral':
                raise ValueError(f'Give a unique, specific audited experiment.name in {path}')
            yield label, 'verified_external_spectral', None, path
    else:
        raise ValueError(group)


def build_plan(args):
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    groups = [s.strip() for s in args.groups.split(',') if s.strip()]
    datasets = [s.strip() for s in args.datasets.split(',') if s.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be nonempty and unique')
    if not groups or len(groups) != len(set(groups)):
        raise ValueError('Groups must be nonempty and unique')
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError('Datasets must be nonempty and unique')
    if args.epochs <= 0 or args.image_size <= 0:
        raise ValueError('epochs and image_size must be positive')
    exts = [Path(x.strip()).resolve() for x in args.external_configs.split(',') if x.strip()]
    jobs=[]; ids=set()
    for dataset in datasets:
        for group in groups:
            for label, model, alpha, source in models_for_group(group, exts):
                template = load_config(source)
                for seed in seeds:
                    variant = f'{group}_{label}'
                    run_id=f'{dataset}/{group}/{variant}/seed_{seed}'
                    if run_id in ids:
                        raise ValueError(f'Duplicate run {run_id}')
                    ids.add(run_id)
                    root = Path(args.output_root).resolve() / dataset / group / variant / f'seed_{seed}'
                    config_path = root / 'reproducible_config.yaml'
                    cfg = deepcopy(template)
                    cfg.setdefault('experiment', {})['name'] = variant
                    cfg['experiment']['output_root'] = str(root)
                    cfg['model']['name'] = model
                    if alpha is not None:
                        cfg['model']['fourier_alpha'] = float(alpha)
                        cfg['model']['fourier_alpha_start'] = float(alpha)
                    cfg.setdefault('train', {})['seed'] = seed
                    cfg['train']['epochs'] = args.epochs
                    if str(cfg['train'].get('scheduler', '')).lower() == 'cosine':
                        cfg['train']['t_max'] = args.epochs
                    cfg.setdefault('data', {})['dataset'] = dataset
                    cfg['data']['root'] = str(Path(args.data_root).resolve())
                    cfg['data']['image_size'] = args.image_size
                    cfg['data']['num_workers'] = args.num_workers
                    if args.batch_size is not None:
                        cfg['data']['batch_size'] = args.batch_size
                    train = [sys.executable, str(ROOT/'scripts/train_one.py'), '--model',model,
                             '--config',str(config_path), '--dataset',dataset,
                             '--data-root',str(Path(args.data_root).resolve()),
                             '--seed',str(seed), '--epochs',str(args.epochs),
                             '--image-size',str(args.image_size), '--num-workers',str(args.num_workers),
                             '--device',args.device,'--output-root',str(root)]
                    ckpt = root / variant / 'checkpoints' / 'best.pt'
                    output = root / 'metrics' / args.split
                    evaluate = [sys.executable,str(ROOT/'scripts/revision/evaluate_revision.py'),
                                '--checkpoint',str(ckpt), '--dataset',dataset,
                                '--data-root',str(Path(args.data_root).resolve()),
                                '--split',args.split,'--image-size',str(args.image_size),
                                '--batch-size',str(args.eval_batch_size),
                                '--num-workers',str(args.num_workers),
                                '--device',args.device,'--output-dir',str(output)]
                    if args.save_predictions:
                        evaluate.append('--save-predictions')
                    jobs.append({'id':run_id, 'group':group,'dataset':dataset,
                                 'seed':seed,'variant':variant,'model':model,
                                 'config_path':str(config_path),'config':cfg,
                                 'source_config':str(source),
                                 'checkpoint':str(ckpt),'metrics_dir':str(output),
                                 'train_command':train,'eval_command':evaluate})
    return jobs


def check_splits(data_root, datasets, split):
    # Fail closed before starting lengthy training; prevent accidental generic
    # train.txt fallback in the reviewer-controlled protocol.
    from scripts.revision.audit_splits import manifest_path
    for dataset in datasets:
        for name in ('train', 'val', split):
            manifest_path(data_root, dataset, name, strict_dataset_paths=True)
        seen = {}
        for name in ('train','val','test'):
            path = manifest_path(data_root,dataset,name,strict_dataset_paths=True)
            ids={line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')}
            if not ids:
                raise ValueError(f'Empty {name} split for {dataset}')
            if seen.keys() & ids:
                raise ValueError(f'Overlapping split IDs for {dataset}')
            for sid in ids:
                if sid in seen:
                    raise ValueError(f'Duplicate sample {sid} in {seen[sid]}/{name}')
                seen[sid]=name


def write_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    text = yaml.safe_dump(cfg,sort_keys=False)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Existing experiment configuration differs: {path}. Choose a new output root.')
    path.write_text(text, encoding='utf-8')


def execute(jobs, args):
    if args.run:
        check_splits(args.data_root, {j['dataset'] for j in jobs}, args.split)
    for job in jobs:
        ckpt=Path(job['checkpoint'])
        result=Path(job['metrics_dir'])/'per_image.csv'
        print('[JOB]',job['id'],flush=True)
        if not args.run:
            print('[PLAN TRAIN]', ' '.join(job['train_command']))
            print('[PLAN EVAL] ', ' '.join(job['eval_command']))
            continue
        write_config(Path(job['config_path']),job['config'])
        run_dir=Path(job['config_path']).parent
        expected={'run_id':job['id'],'model':job['model'],'seed':job['seed'],
                  'dataset':job['dataset'],'config_sha256':sha256_file(job['config_path'])}
        meta_file=run_dir/'experiment_manifest.json'
        if meta_file.is_file() and json.loads(meta_file.read_text(encoding='utf-8')) != expected:
            raise ValueError(f'Existing provenance differs for {job["id"]}')
        write_json(expected,meta_file)
        if not (args.resume and ckpt.is_file()):
            subprocess.run(job['train_command'],check=True,cwd=ROOT)
        else:
            print('[SKIP TRAIN; VALID CHECKPOINT EXISTS]',ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f'Training did not create expected checkpoint: {ckpt}')
        if not (args.resume and result.is_file() and (result.parent/'summary.json').is_file()):
            subprocess.run(job['eval_command'],check=True,cwd=ROOT)
        else:
            print('[SKIP EVAL; OUTPUT EXISTS]',result)
        if not result.is_file():
            raise RuntimeError(f'Missing per-image evaluation: {result}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',default='factorial,residual,convergence',help='factorial,residual,convergence,official')
    p.add_argument('--datasets',default='kvasir_seg')
    p.add_argument('--seeds',default='42,1337,2024')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--data-root',default='data')
    p.add_argument('--output-root',default='outputs_revision/final')
    p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--eval-batch-size',type=int,default=4)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--device',default='auto')
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--external-configs',default='',help='Comma-separated completed official baseline YAML files')
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--resume',action='store_true',help='Skip completed checkpoints/evaluations; never skip absent files')
    p.add_argument('--run',action='store_true',help='Execute; without this flag only write/print an experiment plan')
    args=p.parse_args()
    jobs=build_plan(args)
    output=Path(args.output_root).resolve()
    summary=[{k:v for k,v in j.items() if k!='config'} for j in jobs]
    write_json({'note':'Planning is not proof of trained performance','run_count':len(jobs),'jobs':summary},
               output/'revision_plan.json')
    print(f'Prepared {len(jobs)} jobs: {output / "revision_plan.json"}')
    execute(jobs,args)

if __name__=='__main__':
    main()
