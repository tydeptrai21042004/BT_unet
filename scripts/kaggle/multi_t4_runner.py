#!/usr/bin/env python3
"""Kaggle Tesla T4 sequential experiment runner for BT_unet.

Use ONLY from a checked-out BT_unet GitHub repository. The standalone Kaggle
bootstrap clones the repository, pins optional SHA and calls this module.
Each full experiment yields a separate results ZIP and advances the resume
index only after training, evaluation, and successful archival.
"""
from __future__ import annotations
import argparse
import os, sys, time, json, base64, zlib, hashlib, zipfile, subprocess, shutil, signal, traceback, csv
from pathlib import Path

parser = argparse.ArgumentParser(description='Run BT_unet experiment queue under a Kaggle 12h T4 budget')
parser.add_argument('--scope', choices=['CORE','OPTIONAL'], default='CORE')
parser.add_argument('--start-index', type=int, default=0)
parser.add_argument('--plan-only', action='store_true')
parser.add_argument('--strict-baselines', action='store_true', help='Stop rather than run specialized paper-aligned reimplementations')
args = parser.parse_args()
SCOPE = args.scope
START_INDEX = args.start_index
PLAN_ONLY = args.plan_only
ALLOW_PAPER_ALIGNED_REIMPLEMENTATIONS = not args.strict_baselines

REPO_URL = 'https://github.com/tydeptrai21042004/BT_unet.git'
REPO_ROOT = Path(__file__).resolve().parents[2]
START = float(os.environ.get('BT_T4_MONOTONIC_START', time.monotonic()))
if START > time.monotonic() + 1 or START < time.monotonic() - 24*3600:
    raise ValueError('Invalid BT_T4_MONOTONIC_START')
DEADLINE = START + 11*3600 + 10*60
CLEANUP_RESERVE = 20*60
BASE = Path(os.environ.get('BT_KAGGLE_WORKDIR', '/kaggle/working' if Path('/kaggle/working').exists() else '/tmp/bt_kaggle_test'))
SPEC={'id':'_bootstrap'}
CELL_DIR = BASE/'bt_cell_bootstrap'
REPO = REPO_ROOT
DATA = CELL_DIR/'data'
RESULTS = CELL_DIR/'results'
ARCHIVE = BASE/'BT_UNET_RESULTS__bootstrap.zip'
RUN_META = {'spec':SPEC,'repo_url':REPO_URL,'status':'initialized',
    'utc_created':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}

def remaining():
    return DEADLINE-time.monotonic()

def hard_check(extra=0):
    if remaining() < CLEANUP_RESERVE+extra:
        raise TimeoutError('Session wall-clock budget reached. No fake result will be produced.')

def run_cmd(cmd, *, cwd=None, max_seconds=None, name='command', env=None):
    hard_check()
    log=RESULTS/'logs'/ (name+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    limit=max(20,int(remaining()-CLEANUP_RESERVE))
    if max_seconds is not None:limit=min(limit,int(max_seconds))
    print('\n[RUN]',name,' '.join(map(str,cmd)), 'timeout',limit,'s',flush=True)
    t0=time.monotonic()
    last_heartbeat=t0
    with log.open('w',encoding='utf-8',errors='replace') as fh:
        proc=subprocess.Popen(list(map(str,cmd)),cwd=str(cwd) if cwd else None,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            ret=proc.poll()
            if ret is not None:break
            if time.monotonic()-last_heartbeat >= 120:
                print('[ACTIVE]',name,'elapsed_min',round((time.monotonic()-t0)/60,1),
                      'session_remaining_h',round(max(0,remaining())/3600,2),
                      'log',str(log),flush=True)
                last_heartbeat=time.monotonic()
            if (time.monotonic()-t0)>limit or remaining()<=CLEANUP_RESERVE:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=30)
                raise TimeoutError(f'{name} exceeded its time budget. See {log}. Training must be rerun; this code does not falsely claim mid-epoch resume.')
            time.sleep(2)
    tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-14:]
    for line in tail:print(' ',line[-220:],flush=True)
    if ret:
        print('[FAILED]',name,'exit',ret,'seconds',round(time.monotonic()-t0,1),flush=True)
        raise RuntimeError(f'{name} exit={ret}; see {log}')
    print('[DONE]',name,'seconds',round(time.monotonic()-t0,1),flush=True)
    return time.monotonic()-t0


def install_code():
    """Use the already-cloned repository. Never replace baseline modules in memory."""
    hard_check()
    required = [
        'scripts/train_one.py', 'scripts/revision/evaluate_revision.py',
        'scripts/revision/run_full_revision.py',
        'scripts/revision/audit_baseline_fidelity.py',
        'scripts/prepare_dataset.py', 'scripts/make_splits.py',
    ]
    for path in required:
        if not (REPO/path).is_file():
            raise FileNotFoundError('Missing required GitHub file: '+str(REPO/path))
    pinned = os.environ.get('BT_UNET_COMMIT','').strip().lower()
    if not (REPO/'.git').exists():
        raise RuntimeError('Run from real GitHub checkout; missing .git in '+str(REPO))
    RUN_META['git_commit'] = subprocess.check_output(
        ['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    RUN_META['git_remote'] = subprocess.check_output(
        ['git','-C',str(REPO),'remote','get-url','origin'],text=True).strip()
    if RUN_META['git_remote'].rstrip('/').lower() != REPO_URL.rstrip('/').lower():
        raise RuntimeError('Not the requested BT_unet GitHub remote')
    if pinned and not RUN_META['git_commit'].lower().startswith(pinned):
        raise RuntimeError('Git commit mismatch: rerun with same pinned commit')
    RUN_META['source_mode']='real_github_clone'
    model_paths=[str(p.relative_to(REPO)) for p in (REPO/'src/models').rglob('*.py')]
    tracked=sorted(set(required+model_paths))
    RUN_META['source_code_hashes']={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()
        for p in tracked if (REPO/p).is_file() }
    RUN_META['fidelity_patch_sha256']='already_in_git_no_runtime_patch'
    run_cmd([sys.executable,'scripts/revision/audit_baseline_fidelity.py',
       '--output',str(RESULTS/'baseline_fidelity_audit.json'),'--strict-structure'],
       cwd=REPO,name='local_baseline_fidelity_audit',max_seconds=90)
    if not os.environ.get('BT_KAGGLE_SKIP_PIP'):
        run_cmd([sys.executable,'-m','pip','install','-q','-r','requirements.txt','-r','requirements/revision.txt'],
            cwd=REPO,max_seconds=1400,name='pip_install')
    import torch
    RUN_META['torch']=torch.__version__
    RUN_META['cuda']=bool(torch.cuda.is_available())
    RUN_META['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'
    if not torch.cuda.is_available() and not os.environ.get('BT_KAGGLE_TEST_CPU'):
        raise RuntimeError('Enable GPU T4 in Kaggle Settings -> Accelerator')
    print('[GPU]',RUN_META['gpu'],flush=True)

def baseline_gate():
    # This labels what is actually trained, not what a paper title might imply.
    name=SPEC['model']
    specialized={'pranet','acsnet','hardnet_mseg','cfanet','polyp_pvt','caranet','hsnet','resunetpp'}
    architectural={'unet','unetpp','attention_unet'}
    if name in specialized:
        RUN_META['baseline_fidelity']='paper_aligned_reimplementation_NOT_official_source_verified'
        if os.environ.get('BT_ALLOW_PAPER_ALIGNED','')!='1':
            raise RuntimeError('This specialized baseline is a PAPER-ALIGNED reimplementation, '+
              'NOT an independently verified original-paper model. To run it as an explicitly '+
              'labeled controlled comparison only, set BT_ALLOW_PAPER_ALIGNED=1. '+
              'Never describe these numbers as author-official reproduction.')
    elif name in architectural:
        RUN_META['baseline_fidelity']='architecture_control_NOT_byte_identical_official_code'
    elif name.startswith('dapr_'):
        RUN_META['baseline_fidelity']='BT_unet_repository_proposed_or_ablation_model'
    else:
        raise RuntimeError('Unknown or unaudited model '+name+'; no silent substitution permitted')
    print('[BASELINE_FIDELITY]',name,RUN_META['baseline_fidelity'])


def _find_valid_kvasir_input(base):
    """Use complete original Kvasir-SEG image-mask pairs, not an unrelated split."""
    from scripts.prepare_kvasir_seg import _find_dataset_root, _collect_pairs
    from src.datasets.kvasir_seg_dataset import _resolve_image_mask_dirs, canonical_sample_id

    base=Path(base)
    if not base.exists():
        return None
    root=_find_dataset_root(base,'kvasir_seg')
    if root is not None:
        folders=_resolve_image_mask_dirs(root)
        if folders is not None:
            pairs=_collect_pairs(folders.image_dir,folders.mask_dir)
            if len(pairs)!=1000:
                raise RuntimeError(f'Kvasir-SEG source has {len(pairs)} matched pairs (expected 1000): {root}')
            # Record a content fingerprint of the exact input in this research session.
            digest=hashlib.sha256()
            for sid,img,mask in pairs:
                for tag,filepath in (('I',img),('M',mask)):
                    digest.update((tag+sid+'\0').encode('utf-8'))
                    with filepath.open('rb') as handle:
                        for chunk in iter(lambda:handle.read(1024*1024),b''):
                            digest.update(chunk)
            return ('--source-dir',str(root),digest.hexdigest())
    # Some Kaggle dataset packages expose the original ZIP instead of folders.
    for archive in sorted(base.rglob('*.zip')):
        if not zipfile.is_zipfile(archive):
            continue
        with zipfile.ZipFile(archive) as z:
            images=set()
            masks=set()
            for name in z.namelist():
                p=Path(name)
                if p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
                    continue
                tokens=[x.lower() for x in p.parts[:-1]]
                sid=canonical_sample_id(p.stem)
                if 'images' in tokens:
                    images.add(sid)
                elif 'masks' in tokens:
                    masks.add(sid)
            if len(images)==1000 and images==masks:
                digest=hashlib.sha256()
                with archive.open('rb') as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b''):
                        digest.update(chunk)
                return ('--zip-path',str(archive),digest.hexdigest())
    return None


def _resolve_kvasir_kaggle():
    """Use Kaggle's dataset delivery instead of disabling Simula TLS checks."""
    mounted=Path('/kaggle/input')
    if mounted.is_dir():
        for folder in sorted(mounted.iterdir()):
            if 'kvasir' in folder.name.lower():
                result=_find_valid_kvasir_input(folder)
                if result:
                    print('[DATA] Using existing Kaggle input:',result[1],flush=True)
                    return result, 'kaggle_input_mount'
    try:
        import kagglehub
    except ImportError:
        run_cmd([sys.executable,'-m','pip','install','-q','kagglehub'],
                name='install_kagglehub',max_seconds=240)
        import kagglehub
    handle='debeshjha1/kvasirseg'
    print('[DATA] Retrieving Kvasir-SEG from Kaggle:',handle,flush=True)
    try:
        # Bound network I/O to protect the 12-hour session deadline.
        marker=RESULTS/'kagglehub_dataset_path.txt'
        snippet=("from pathlib import Path; import kagglehub; "
                 +"p=kagglehub.dataset_download('debeshjha1/kvasirseg'); "
                 +"Path("+repr(str(marker))+").write_text(str(p),encoding='utf-8')")
        run_cmd([sys.executable,'-c',snippet],name='kagglehub_kvasir_fetch',max_seconds=1800)
        path=Path(marker.read_text(encoding='utf-8').strip())
    except Exception as exc:
        raise RuntimeError('Kaggle dataset retrieval failed. Add the public Kvasir-SEG '
            'dataset (debeshjha1/kvasirseg) under Kaggle Add Input, then rerun. '
            'You may also provide BT_DATA_KVASIR_SEG_SOURCE or BT_DATA_KVASIR_SEG_ZIP.') from exc
    result=_find_valid_kvasir_input(path)
    if not result:
        raise RuntimeError('Kaggle Kvasir-SEG download did not contain the expected '
            '1000 paired images and masks: '+str(path))
    print('[DATA] Verified 1000 image/mask pairs from:',result[1],flush=True)
    return result, 'kagglehub:'+handle


def setup_datasets():
    sys.path.insert(0,str(REPO))
    from scripts.revision.audit_splits import audit
    datasets=set(SPEC.get('sources',[])+SPEC.get('targets',[])+[SPEC['dataset']])
    for dataset in sorted(datasets):
        if dataset in PREPARED:
            if dataset in DATASET_META:
                for key, value in DATASET_META[dataset].items():
                    RUN_META.setdefault(key,{})[dataset]=__import__('copy').deepcopy(value)
            continue
        args=[sys.executable,'scripts/prepare_dataset.py','--dataset',dataset,'--data-root',str(DATA),'--image-size',str(SPEC['image_size'])]
        source_path=os.getenv('BT_DATA_'+dataset.upper()+'_SOURCE')
        zip_path=os.getenv('BT_DATA_'+dataset.upper()+'_ZIP')
        if source_path:
            args+=['--source-dir',source_path]
            RUN_META.setdefault('dataset_provenance',{})[dataset]={'source':'explicit_source_dir','path':source_path}
        elif zip_path:
            args+=['--zip-path',zip_path]
            RUN_META.setdefault('dataset_provenance',{})[dataset]={'source':'explicit_zip','path':zip_path}
        elif dataset=='kvasir_seg':
            (flag,path,sha),where=_resolve_kvasir_kaggle()
            args += [flag,path]
            RUN_META.setdefault('dataset_provenance',{})[dataset]={
                'source':where,'path':path,'sha256_of_input':sha,
                'original_image_mask_pairs':1000}
        run_cmd(args,cwd=REPO,max_seconds=3600,name='prepare_'+dataset)
        if dataset=='kvasir_seg':
            metadata=DATA/'processed'/'kvasir_seg'/'metadata.csv'
            if not metadata.is_file():
                raise FileNotFoundError('Kvasir metadata missing after preparation: '+str(metadata))
            with metadata.open(newline='',encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            if len(rows)!=1000:
                raise RuntimeError('Expected 1000 prepared Kvasir-SEG pairs, got '+str(len(rows)))
            print('[DATA] Prepared Kvasir-SEG: 1000 image-mask pairs',flush=True)
        split_args=[sys.executable,'scripts/make_splits.py','--dataset',dataset,'--data-root',str(DATA),'--image-size',str(SPEC['image_size']),'--seed','42']
        if dataset=='isbi2012':split_args+=['--strategy','contiguous']
        run_cmd(split_args,cwd=REPO,max_seconds=600,name='split_'+dataset)
        # Optional externally reviewed official / patient-level manifests. If present,
        # overwrite the exploratory image-level splits for every session identically.
        split_dir=os.environ.get('BT_SPLITS_DIR')
        if split_dir:
            from_dir=Path(split_dir)/dataset
            if not from_dir.is_dir():raise FileNotFoundError('Missing grouped split folder: '+str(from_dir))
            for part in ('train','val','test'):
                src=from_dir/(part+'.txt')
                if not src.is_file():raise FileNotFoundError('Missing grouped split: '+str(src))
                shutil.copyfile(src,DATA/'splits'/dataset/(part+'.txt'))
            RUN_META.setdefault('split_protocol',{})[dataset]='user_supplied_grouped_or_official'
        else:
            RUN_META.setdefault('split_protocol',{})[dataset]='exploratory_random_image_split_seed42'
        # Strict manifest existence + overlap checks, no implicit fallback.
        for part in ['train','val','test']:
            p=DATA/'splits'/dataset/(part+'.txt')
            if not p.is_file() or not p.read_text().strip():raise RuntimeError('Missing nonempty dataset-specific split: '+str(p))
        PREPARED.add(dataset)
        DATASET_META[dataset]={key:__import__('copy').deepcopy(RUN_META.get(key,{}).get(dataset))
                            for key in ('dataset_provenance','split_protocol')}
    sys.path.insert(0,str(REPO))
    from scripts.revision.run_full_revision import check_splits
    check_splits(DATA,datasets,'test')
    # Dataset-level exact duplicate audit for cross-domain training.
    cross_key=(tuple(SPEC.get('sources',[])),tuple(SPEC.get('targets',[])))
    if SPEC['type']=='cross' and cross_key not in CROSS_CHECKED:
        from src.datasets import build_dataset
        import hashlib
        hashes={}
        for source in SPEC['sources']:
            for split in ['train','val']:
                for sid,img,_ in build_dataset(source,str(DATA),split=split,image_size=SPEC['image_size']).samples:
                    digest=hashlib.sha256(Path(img).read_bytes()).hexdigest()
                    hashes.setdefault(digest,(source,split,sid))
        for target in SPEC['targets']:
            for sid,img,_ in build_dataset(target,str(DATA),split='test',image_size=SPEC['image_size']).samples:
                digest=hashlib.sha256(Path(img).read_bytes()).hexdigest()
                if digest in hashes:raise RuntimeError('Data leakage! '+str((hashes[digest],target,sid)))
        CROSS_CHECKED.add(cross_key)

def make_job():
    import yaml
    t=SPEC['type']
    if t=='revision':
        from scripts.revision.run_full_revision import build_plan
        from argparse import Namespace
        arg=Namespace(groups=SPEC['group'],datasets=SPEC['dataset'],seeds=str(SPEC['seed']),epochs=SPEC['epochs'],data_root=str(DATA),output_root=str(RESULTS/'jobs'),image_size=SPEC['image_size'],batch_size=SPEC['batch_size'],eval_batch_size=4,num_workers=SPEC['num_workers'],device='cuda:0' if not os.environ.get('BT_KAGGLE_TEST_CPU') else 'cpu',split='test',external_configs='',save_predictions=True)
        options=build_plan(arg)
        selected=[j for j in options if j['variant']==SPEC['group']+'_'+SPEC['variant']]
        if len(selected)!=1:raise RuntimeError('Variant not uniquely found in revision run plan')
        job=selected[0]
        config_path=Path(job['config_path']);config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(yaml.safe_dump(job['config'],sort_keys=False),encoding='utf-8')
        return job['train_command'],Path(job['checkpoint']),[('test',job['eval_command'])],job['config'],config_path
    src=REPO/SPEC['config']
    if not src.is_file():raise FileNotFoundError('Missing actual model config: '+str(src))
    cfg=yaml.safe_load(src.read_text())
    if not isinstance(cfg,dict) or 'model' not in cfg:raise ValueError('Malformed config')
    model=SPEC['model'];tag=SPEC['id']
    run_root=RESULTS/'jobs'/tag
    cfg.setdefault('experiment',{})['name']=tag
    cfg['experiment']['output_root']=str(run_root)
    cfg.setdefault('data',{}).update({'dataset':SPEC['dataset'],'root':str(DATA),'image_size':SPEC['image_size'],'batch_size':SPEC['batch_size'],'num_workers':SPEC['num_workers']})
    cfg.setdefault('train',{}).update({'epochs':SPEC['epochs'],'t_max':SPEC['epochs'],'seed':SPEC['seed'],'device':'cpu' if os.environ.get('BT_KAGGLE_TEST_CPU') else 'cuda:0'})
    if t=='cross':
        cfg['data']['train_datasets']=SPEC['sources']
        cfg['data']['val_dataset']=SPEC['sources'][0]
    if t=='lowdata':
        make_dir=SESSION/'common_lowdata_splits'
        if not (make_dir/'train_100pct_seed42.txt').is_file():
            run_cmd([sys.executable,'scripts/revision/make_low_data_splits.py','--data-root',str(DATA),'--dataset',SPEC['dataset'],'--fractions','0.1,0.25,0.5,1.0','--seed','42','--output-dir',str(make_dir)],cwd=REPO,name='lowdata_manifest')
        train_subset=make_dir/('train_%03dpct_seed42.txt'%SPEC['fraction_pct'])
        if not train_subset.is_file():raise FileNotFoundError(train_subset)
        cfg['data']['train_split_file']=str(train_subset)
    config_path=run_root/'reproducible_config.yaml';config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    train=[sys.executable,'scripts/train_one.py','--model',model,'--dataset',SPEC['dataset'],'--config',str(config_path),'--data-root',str(DATA),'--image-size',str(SPEC['image_size']),'--batch-size',str(SPEC['batch_size']),'--epochs',str(SPEC['epochs']),'--seed',str(SPEC['seed']),'--device','cpu' if os.environ.get('BT_KAGGLE_TEST_CPU') else 'cuda:0','--num-workers',str(SPEC['num_workers']),'--output-root',str(run_root)]
    ckpt=run_root/tag/'checkpoints'/'best.pt'
    targets=SPEC.get('targets',[SPEC['dataset']])
    evaluations=[]
    for dataset in targets:
        dst=run_root/'evaluations'/dataset
        cmd=[sys.executable,'scripts/revision/evaluate_revision.py','--checkpoint',str(ckpt),'--dataset',dataset,'--data-root',str(DATA),'--split','test','--batch-size','4','--image-size',str(SPEC['image_size']),'--num-workers',str(SPEC['num_workers']),'--device','cpu' if os.environ.get('BT_KAGGLE_TEST_CPU') else 'cuda:0','--output-dir',str(dst),'--save-predictions']
        evaluations.append((dataset,cmd))
    return train,ckpt,evaluations,cfg,config_path

def calibrate(train,config,ckpt):
    import yaml
    # A complete 1-epoch training run with same architecture, batch, image count and loss.
    # Separate config/output: these artifacts are discarded, never reported as scientific results.
    cal_root=CELL_DIR/'calibration'
    c=__import__('copy').deepcopy(config)
    c['experiment']['output_root']=str(cal_root)
    c['experiment']['name']='calibration_'+SPEC['id']
    c['train']['epochs']=1;c['train']['t_max']=1
    cp=cal_root/'calibration_config.yaml';cp.parent.mkdir(parents=True,exist_ok=True)
    cp.write_text(yaml.safe_dump(c,sort_keys=False))
    cmd=list(train)
    for flag,value in [('--config',str(cp)),('--epochs','1'),('--output-root',str(cal_root))]:
        if flag in cmd:cmd[cmd.index(flag)+1]=value
        else:cmd.extend([flag,value])
    cost=run_cmd(cmd,cwd=REPO,max_seconds=3600,name='calibration_1epoch')
    RUN_META['calibration_seconds']=round(cost,2)
    expected_seconds=cost*SPEC['epochs']*2.0
    RUN_META['estimated_full_train_seconds']=round(expected_seconds,1)
    print('[PREFLIGHT] 1-epoch wall time',round(cost,1),'s; conservative estimate:',round(expected_seconds/3600,2),'h')
    shutil.rmtree(cal_root,ignore_errors=True)
    # Refuse to start an unfinishable run rather than change model epochs silently.
    if expected_seconds>remaining()-CLEANUP_RESERVE-EVAL_AND_OUTPUT_RESERVE:
        raise RuntimeError('PROJECTED_OVER_BUDGET: predicted training plus evaluation/analysis does not fit within T4 session. No epoch count changed.')

def postprocess(ckpt):
    if not SPEC.get('posthoc'):return
    dest=RESULTS/'analysis';dest.mkdir(parents=True,exist_ok=True)
    # Only actual trained checkpoint and real prepared data are analyzed.
    run_cmd([sys.executable,'scripts/revision/analyze_spectrum.py','--checkpoint',str(ckpt),'--output-dir',str(dest/'weights'),'--device','cpu'],cwd=REPO,max_seconds=800,name='spectrum_weights')
    run_cmd([sys.executable,'scripts/revision/analyze_feature_spectrum.py','--checkpoint',str(ckpt),'--dataset',SPEC['dataset'],'--data-root',str(DATA),'--max-images','50','--output-dir',str(dest/'activations'),'--device','cuda:0' if not os.environ.get('BT_KAGGLE_TEST_CPU') else 'cpu'],cwd=REPO,max_seconds=1000,name='spectrum_activations')
    run_cmd([sys.executable,'scripts/revision/run_frequency_interventions.py','--checkpoint',str(ckpt),'--dataset',SPEC['dataset'],'--data-root',str(DATA),'--output-root',str(dest/'frequency_interventions'),'--device','cuda:0' if not os.environ.get('BT_KAGGLE_TEST_CPU') else 'cpu'],cwd=REPO,max_seconds=1800,name='frequency_interventions')
    run_cmd([sys.executable,'scripts/revision/run_robustness.py','--checkpoint',str(ckpt),'--dataset',SPEC['dataset'],'--data-root',str(DATA),'--output-root',str(dest/'corruptions'),'--corruptions','clean,gaussian_noise,gaussian_blur,motion_blur,glare,low_contrast,dark,jpeg','--severities','1,2,3','--device','cuda:0' if not os.environ.get('BT_KAGGLE_TEST_CPU') else 'cpu'],cwd=REPO,max_seconds=2400,name='robustness')

def package_all():
    if ARCHIVE.exists():
        raise FileExistsError('Archive already exists; refusing to overwrite prior results: '+str(ARCHIVE))
    RESULTS.mkdir(parents=True,exist_ok=True)
    RUN_META['elapsed_seconds']=round(time.monotonic()-JOB_START,1)
    RUN_META['total_session_elapsed_seconds']=round(time.monotonic()-START,1)
    (RESULTS/'session_manifest.json').write_text(json.dumps(RUN_META,indent=2,default=str)+'\n')
    # Reduce archive size while retaining both best and last trained model weights.
    # Optimizer state is intentionally omitted from exported checkpoints: the
    # original trainer cannot truly resume an interrupted multi-epoch run.
    try:
        import torch
        slimmed=[]
        for checkpoint in RESULTS.rglob('*.pt'):
            blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if isinstance(blob,dict) and 'optimizer' in blob:
                blob.pop('optimizer',None)
                temp=checkpoint.with_suffix('.tmp')
                torch.save(blob,temp)
                temp.replace(checkpoint)
                slimmed.append(str(checkpoint.relative_to(RESULTS)))
        RUN_META['checkpoints_without_optimizer']=slimmed
        (RESULTS/'session_manifest.json').write_text(json.dumps(RUN_META,indent=2,default=str)+'\n')
    except Exception as e:
        print('[WARNING] Could not minimize checkpoint archive:',e)
    with zipfile.ZipFile(ARCHIVE,'w',allowZip64=True) as z:
        for p in RESULTS.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(RESULTS)
            z.write(p,arcname=str(Path('session_results')/SPEC['id']/rel),compress_type=zipfile.ZIP_STORED if p.suffix in {'.pt','.pth','.png','.jpg','.zip'} else zipfile.ZIP_DEFLATED)
    print('\n'+'='*60)
    print('RESULT ZIP:',ARCHIVE)
    print('SIZE MB:',round(ARCHIVE.stat().st_size/1e6,2),'STATUS:',RUN_META['status'])
    print('='*60,flush=True)


# Compressed original immutable experiment plans; 129 CORE and 132 OPTIONAL jobs.
QUEUE_B64_BY_SCOPE = {'CORE': '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', 'OPTIONAL': 'eNrV2uFu2jAUBeB3yW+00kA3iVeZpsgkDkSFJIrTVmzauy+oDFFNi69976XH/6rWmH7SqXRyyvdf2daM5b5wzU+bbb4usrJr62aXbS5fuIfaNMNDZfqheGnt+OVkjodskVVmNM6O53OvZbFaLqfv2b4r9y7bPC0XWVNNP3KjaSszVEVxOVQU14uKwj2uVt+mlzVHs7OXX2D1lC+yY1fZw/Ty69npUPtyLN664dkO0xtMZ5y10zucb1hk46m3N++W/V58pipf5mue6nwDmmqd80zrnCEarDtf3vdc0fUiQvquZ7XSp6Lypc+vYqZPRTWfPr+JlT4ZDjl3uqETxvjipps1Ycx8yrQjJsEgxkszXEIMSrA0YyXE8EdKJ1CeDlB2h66ttn7L5WBavY2nQ+9vPB1yjwuXpdTneDr0XsfTwfa7cFYCPY+Bwu17DBRi74vjwPa/aA5mD4zmAPZBOzZuHnI+kU4DjPEgd74YD2rLo1pS6XUxHuQmF+OB7G5UCHhbC2Zg9rNgBlojCwFAdrBAAF7rCgQA9qzGbZt8+ZjPK/6eSqdvcVzIvYvjQu1foaZUehjHhdzHOC7IXhYKAu9n0RzMnhbNQetrMRDI3hYJwetvkRDAHvf8alwzFE3rxuHlaNtxnvTP8XSanawUuevJSlHbH1+ZSh+UlSI3RFkpZGfkE8FbpCAQs1cKAtGapgwNsnuK0fDaqBjts/qpKZ3gP6ffbyOE8P2gVgyVUL74eVDMACqh5oPnIfGiN47TX0DTtaKfUfx4KyWKH16gFsn7YL0RpWG5Ub0P1hNdGpUV4dIMRlB4uY4Q2stJrbSquXz59LmYwVRzzUfRp+JlsBYl1dQE1qoB1EF541erpk8H5clerRe9/XT6/Dscnd1JsW7vJMTw9rhWGO/A9AWTxGTG8w7M+aiSkLzASlboPfFZZ6/5qKMi8qZR8zlHReQJnt5DTt8dTn3Rv4qJrhcSknc9q5U+VZ0vhX4dM4mquvlE+m28VIrW4J761NKrPrQoobwxVH1iUUJ50qf4uPK/GYv6uUzwiTGYgTkqBjNwZ0QyJc3hMJqXxlQYzQMeB6km/DkwXII6AIZLACc/MgJ75AtmYM56wQzUIY8KSW66i4Thj3WRMNB5jqwBHuRCDYgTXKgBcXSjGlKZ2WI8yMNajAdySiNDsMezYAbmXBbMwBvI+B81BJ/OBIGYo5ogEHduE0CmOcQpwNOY6BTgwOMdX4s/60kaUQc/SSPgFCjAwx4JBYGY86EgEHVY5BOTmxzFyfhjpDgZdKYUcAIPmHI6xGlTToc4evJ1qcyhslLkoVRWCjmhChCxx1VBIObsKgiUH2R//AEDzX54'}
QUEUE_SHA256_BY_SCOPE = {'CORE': '18988d518b31755ce98920625ca60183b55251881eb94f376fd12976506e24ef', 'OPTIONAL': '24c37d4f27a09e4df6e889f60551a73f6141ed3923540d3f0c281b8f05097ad2'}
QUEUE = json.loads(zlib.decompress(base64.b64decode(QUEUE_B64_BY_SCOPE[SCOPE])))
QUEUE_SHA256 = QUEUE_SHA256_BY_SCOPE[SCOPE]
QUEUE_DATA_CANONICAL = json.dumps(QUEUE,sort_keys=True,separators=(',',':')).encode()
assert hashlib.sha256(QUEUE_DATA_CANONICAL).hexdigest() == QUEUE_SHA256, "Queue plan digest mismatch"
assert len(QUEUE) == {'CORE':129,'OPTIONAL':132}[SCOPE]
assert len({x['id'] for x in QUEUE}) == len(QUEUE)
assert all(x['epochs'] in (30,50,100) and x['batch_size']==6 and x['image_size']==352 for x in QUEUE)

PREPARED = set()
DATASET_META = {}
CROSS_CHECKED = set()
CALIBRATION_CACHE = {}
JOB_START = START
EVAL_AND_OUTPUT_RESERVE = 1800
SESSION = BASE / ('BT_UNET_MULTI_' + SCOPE + '_from_' + str(START_INDEX))
SESSION_LOG = SESSION/'session_status.json'
SESSION_REPO = REPO_ROOT
SESSION_DATA = SESSION/'data'
SESSION_RESULTS = SESSION/'job_outputs'
SESSION_LOGS = SESSION/'session_logs'
SOURCE_META = {}

def _activate(spec):
    global SPEC, CELL_DIR, REPO, DATA, RESULTS, ARCHIVE, RUN_META, JOB_START
    SPEC = spec
    JOB_START=time.monotonic()
    CELL_DIR=SESSION/'job_work'/SPEC['id']
    REPO=SESSION_REPO
    DATA=SESSION_DATA
    RESULTS=SESSION_RESULTS/SPEC['id']
    ARCHIVE=BASE/('BT_UNET_RESULTS__'+SPEC['id']+'.zip')
    RUN_META={'spec':SPEC,'repo_url':REPO_URL,'status':'initialized',
              'utc_created':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              'plan_sha256':QUEUE_SHA256,'scope':SCOPE}
    RUN_META.update(__import__('copy').deepcopy(SOURCE_META))
    CELL_DIR.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)

def _calibration_key(config):
    # Eliminate experiment-specific name/paths, seed, and total epochs only;
    # retain architecture, optimizer, dataset, train subset, and every ablation parameter.
    c=__import__('copy').deepcopy(config)
    c.get('experiment',{}).pop('name',None)
    c.get('experiment',{}).pop('output_root',None)
    for key in ('seed','epochs','t_max'):
        c.get('train',{}).pop(key,None)
    c.get('data',{}).pop('root',None)
    identity=json.dumps(c,sort_keys=True,default=str,separators=(',',':')).encode()
    return hashlib.sha256(identity).hexdigest()

def _preflight(train, config, ckpt, evaluations):
    global EVAL_AND_OUTPUT_RESERVE
    # A realistic conservative bound includes *all* test targets and optional post-hoc GPU work.
    EVAL_AND_OUTPUT_RESERVE=600 + 1200*len(evaluations) + (6800 if SPEC.get('posthoc') else 0)
    key=_calibration_key(config)
    if key in CALIBRATION_CACHE:
        cost=CALIBRATION_CACHE[key]
        RUN_META['calibration_reused']=True
        RUN_META['calibration_seconds']=round(cost,2)
        print('[REUSED 1-EPOCH CALIBRATION]',round(cost,1),'s for same config ignoring seed',flush=True)
    else:
        # calibrate() has a budget check after measuring. On failure, the time observation is still logged.
        calibrate(train, config, ckpt)
        cost=RUN_META['calibration_seconds']
        CALIBRATION_CACHE[key]=cost
    expected=cost*SPEC['epochs']*2.0
    RUN_META['estimated_full_train_seconds']=round(expected,1)
    RUN_META['evaluation_and_archiving_reserve_seconds']=EVAL_AND_OUTPUT_RESERVE
    print('[BUDGET]',SPEC['id'],'expected GPU train',round(expected/3600,2),'h',
          'eval/analysis & output reserve',round(EVAL_AND_OUTPUT_RESERVE/3600,2),'h',
          'remaining',round(remaining()/3600,2),'h',flush=True)
    if expected + EVAL_AND_OUTPUT_RESERVE + CLEANUP_RESERVE > remaining():
        raise RuntimeError('PROJECTED_OVER_BUDGET: defer this job to another T4 session; no partial training starts.')
    return expected

def _save_queue(status):
    SESSION_LOG.parent.mkdir(parents=True,exist_ok=True)
    status['session_elapsed_seconds']=round(time.monotonic()-START,1)
    status['remaining_seconds']=round(remaining(),1)
    status['next_start_index']=status['next_index']
    status['plan_sha256']=QUEUE_SHA256
    tmp=SESSION_LOG.with_suffix('.tmp')
    tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str)+'\n')
    tmp.replace(SESSION_LOG)
    (BASE/('BT_UNET_NEXT_'+SCOPE+'.json')).write_text(json.dumps({
       'scope':SCOPE,'next_start_index':status['next_index'],'plan_sha256':QUEUE_SHA256,
       'completed_in_this_session':status['completed_ids'],
       'note':'Use NEXT_START_INDEX for next independent Kaggle session. Keep prior result ZIPs.'
    },indent=2)+'\n')

def run_multi_session():
    global SOURCE_META, ARCHIVE
    if not 0 <= START_INDEX <= len(QUEUE):
        raise ValueError('START_INDEX outside job queue')
    if PLAN_ONLY:
        print('PLAN ONLY',SCOPE,'jobs',len(QUEUE),'starting',START_INDEX)
        for j,s in enumerate(QUEUE[START_INDEX:],START_INDEX):
            print(f'{j:03d} {s["id"]} | {s["epochs"]} epochs | {s["model"]} | {s["dataset"]}')
        return
    if START_INDEX == len(QUEUE):
        print('[COMPLETE QUEUE] All planned jobs already indexed; nothing to train.')
        return
    SESSION.mkdir(parents=True,exist_ok=True)
    status={'scope':SCOPE,'start_index':START_INDEX,'next_index':START_INDEX,
       'total_jobs':len(QUEUE),'completed_ids':[],'failed_jobs':[],
       'stop_reason':None,'git_commit':None,'session_archives':[]}
    _activate(QUEUE[START_INDEX])
    # Set up the code exactly once for every grouped Kaggle session.
    try:
        install_code()
        SOURCE_META={k:__import__('copy').deepcopy(RUN_META[k]) for k in (
            'git_commit','git_remote','source_mode','fidelity_patch_sha256',
            'source_code_hashes','torch','cuda','gpu') if k in RUN_META}
        status['git_commit']=SOURCE_META.get('git_commit')
        print('[SHARED REPO AND DEPENDENCIES READY]')
    except BaseException as exc:
        status['stop_reason']='shared_setup_failed: '+str(exc)
        _save_queue(status)
        raise
    _save_queue(status)
    for index in range(START_INDEX,len(QUEUE)):
        spec=QUEUE[index]
        _activate(spec)
        print('\n'+'='*70,flush=True)
        print(f'[JOB {index+1}/{len(QUEUE)}] {spec["id"]}',flush=True)
        print('[SESSION ELAPSED]',round((time.monotonic()-START)/3600,2),'h',flush=True)
        try:
            hard_check(extra=1800)
            if os.environ.get('BT_KAGGLE_TEST_EPOCHS'):
                raise ValueError('Per-model epoch override disabled for scientific comparisons')
            if ALLOW_PAPER_ALIGNED_REIMPLEMENTATIONS:
                os.environ['BT_ALLOW_PAPER_ALIGNED']='1'
            if shutil.disk_usage(BASE).free < 5*1024**3:
                raise RuntimeError('DISK_LOW: insufficient disk space to safely train/package next job')
            baseline_gate()
            setup_datasets()
            train,ckpt,evals,config,config_path=make_job()
            _preflight(train,config,ckpt,evals)
            RUN_META['status']='training'
            # Explicit post-train evaluation/analysis/packaging time is held back.
            train_limit=max(20,remaining()-CLEANUP_RESERVE-EVAL_AND_OUTPUT_RESERVE)
            run_cmd(train,cwd=REPO,max_seconds=train_limit,name='train_full')
            if not ckpt.is_file():
                raise FileNotFoundError('Best checkpoint missing: '+str(ckpt))
            RUN_META['status']='evaluating'
            for label, command in evals:
                run_cmd(command,cwd=REPO,max_seconds=min(1800,remaining()-CLEANUP_RESERVE-700),name='eval_'+label)
            if SPEC.get('posthoc'):
                postprocess(ckpt)
            RUN_META['status']='complete'
            package_all()  # ONE INDEPENDENT ORIGINAL-AGGREGATOR-COMPATIBLE ZIP PER JOB
            # Outputs are safely preserved in the verified ZIP; release unpacked run data.
            shutil.rmtree(RESULTS,ignore_errors=True)
            status['completed_ids'].append(spec['id'])
            status['session_archives'].append(str(ARCHIVE))
            status['next_index']=index+1
            status['stop_reason']='more_jobs_remaining'
            _save_queue(status)
            print('[JOB COMPLETE] Next index:',index+1,flush=True)
            # Keep enough time for a further calibration and emergency finalization.
            if remaining() < CLEANUP_RESERVE+2700:
                status['stop_reason']='remaining_budget_too_small_for_new_job'
                break
        except BaseException as exc:
            reason=str(exc)
            if ('PROJECTED_OVER_BUDGET' in reason or 'DISK_LOW' in reason or isinstance(exc,TimeoutError) and RUN_META.get('status')=='initialized'):
                status['stop_reason']='deferred_for_next_session: '+reason
                print('[DEFERRED] No experiment settings changed:',reason,flush=True)
            else:
                status['stop_reason']='blocked_or_failed: '+reason
                RUN_META['status']='failed'
                RUN_META['error']=reason
                status['failed_jobs'].append({'index':index,'id':spec['id'],'error':reason})
                print('[ERROR]',traceback.format_exc(),flush=True)
                try:
                    ARCHIVE=BASE/('BT_UNET_FAILED__'+spec['id']+'.zip')
                    package_all()
                except Exception as package_exc:
                    print('[FAILED TO ARCHIVE LOGS]',package_exc)
            break
        finally:
            _save_queue(status)
    if status['next_index'] == len(QUEUE):
        status['stop_reason']='all_jobs_completed'
    _save_queue(status)
    print('\n'+'='*75)
    print('SCOPE',SCOPE,'COMPLETED THIS SESSION',len(status['completed_ids']))
    print('NEXT_START_INDEX =',status['next_index'])
    print('REMAINING PLANNED JOBS =',len(QUEUE)-status['next_index'])
    print('STATUS FILE:',SESSION_LOG)
    print('NEXT SESSION STATE:',BASE/('BT_UNET_NEXT_'+SCOPE+'.json'))
    print('RESULTS: each BT_UNET_RESULTS__*.zip in /kaggle/working')
    print('STOP REASON:',status['stop_reason'])
    print('='*75,flush=True)

if __name__=='__main__':
    run_multi_session()
