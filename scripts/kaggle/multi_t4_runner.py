#!/usr/bin/env python3
"""Kaggle Tesla T4 sequential experiment runner for BT_unet.

Use ONLY from a checked-out BT_unet GitHub repository. The standalone Kaggle
bootstrap clones the repository, pins optional SHA and calls this module.
AUTO sessions save one compact, paper-ready ZIP containing all completed
experiment summaries, per-image CSV files and training histories; no checkpoints.
Other scopes retain the original per-job archival format.
The resume counter advances only after full training/evaluation/export.
"""
from __future__ import annotations
import argparse
import os, sys, time, json, base64, zlib, hashlib, zipfile, subprocess, shutil, signal, traceback, csv
from pathlib import Path

parser = argparse.ArgumentParser(description='Run BT_unet experiment queue under a Kaggle 12h T4 budget')
parser.add_argument('--scope', choices=['ALL','CORE','OPTIONAL','AUTO'], default='ALL',
    help='AUTO runs only jobs whose required datasets have public direct HTTPS ZIP URLs; independent 1-based queue')
parser.add_argument('--start-index', type=int, default=None,
    help='Experiment number (1-based when --one-based; else legacy 0-based index)')
parser.add_argument('--one-based', action='store_true',
    help='Use human-readable experiment numbers: first experiment is 1')
parser.add_argument('--bundle-session', action='store_true',
    help='Also make a single ZIP containing every completed independent job ZIP from this session')
parser.add_argument('--max-hours', type=float, default=11.0,
    help='Maximum notebook wall-clock budget, including setup and packaging (default: 11)')
parser.add_argument('--keep-job-zips', action='store_true',
    help='Keep loose job ZIPs after a verified session ZIP (doubles saved storage)')
parser.add_argument('--keep-scratch', action='store_true',
    help='Preserve unpacked dataset/cache after jobs (consumes Kaggle output quota)')
parser.add_argument('--plan-csv', action='store_true',
    help='Write numbered experiment manifest CSV without downloading or training')
parser.add_argument('--plan-only', action='store_true')
parser.add_argument('--strict-baselines', action='store_true', help='Stop rather than run specialized paper-aligned reimplementations')
args = parser.parse_args()
PAPER_METRICS_COMPACT_V2 = True  # Kaggle bootstrap requires the new results-only format
SCOPE = args.scope
if args.start_index is None:
    args.start_index = 1 if args.one_based else 0
if args.one_based and args.start_index < 1:
    parser.error('--start-index must be at least 1 with --one-based')
START_INDEX = args.start_index - 1 if args.one_based else args.start_index
PLAN_ONLY = args.plan_only
ALLOW_PAPER_ALIGNED_REIMPLEMENTATIONS = not args.strict_baselines

REPO_URL = 'https://github.com/tydeptrai21042004/BT_unet.git'
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
START = float(os.environ.get('BT_T4_MONOTONIC_START', time.monotonic()))
if START > time.monotonic() + 1 or START < time.monotonic() - 24*3600:
    raise ValueError('Invalid BT_T4_MONOTONIC_START')
if not 0.25 <= args.max_hours <= 11.0:
    parser.error('--max-hours must be between 0.25 and 11.0')
DEADLINE = START + args.max_hours*3600
CLEANUP_RESERVE = min(20*60, args.max_hours*3600/4)
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


def sha256_path(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

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
    if SCOPE != 'AUTO':
        run_cmd([sys.executable,'scripts/revision/audit_baseline_fidelity.py',
           '--output',str(RESULTS/'baseline_fidelity_audit.json'),'--strict-structure'],
           cwd=REPO,name='local_baseline_fidelity_audit',max_seconds=90)
    else:
        print('[AUTO] Skipping baseline source audit to prioritize training.',flush=True)
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
    elif SPEC.get('type')=='revision' and SPEC.get('group') in {'factorial','residual','convergence','primary'}:
        # Residual controls have labels such as additive_a050; the real architecture
        # is selected and validated by build_plan(), not by this display label.
        RUN_META['baseline_fidelity']='BT_unet_repository_reviewer_control'
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
    """Resolve Kvasir-SEG without Kaggle's disallowed non-interactive attach API.

    Source order: a previously attached Kaggle Input; Simula's public HTTPS ZIP;
    an authenticated Kaggle CLI download (optional). Explicit data source/ZIP
    overrides are handled by setup_datasets() before this helper is called.
    Every source must contain exactly 1,000 matched image/mask pairs.
    """
    from src.datasets.registry import get_dataset_spec

    mounted = Path(os.environ.get('BT_KAGGLE_INPUT_ROOT', '/kaggle/input'))
    if mounted.is_dir():
        # A mounted input might have a different user-chosen slug, but scanning
        # only for 'kvasir' is cheap and prevents choosing an unrelated dataset.
        candidates = [p for p in sorted(mounted.iterdir())
                      if 'kvasir' in p.name.lower()]
        for folder in candidates:
            result = _find_valid_kvasir_input(folder)
            if result:
                print('[DATA] Verified Kvasir-SEG from attached Kaggle Input:',
                      result[1], flush=True)
                return result, 'kaggle_input_mount'

    failures = []
    download_dir = SESSION_DATA / 'downloads' / 'kvasir_seg'
    download_dir.mkdir(parents=True, exist_ok=True)
    url = get_dataset_spec('kvasir_seg').default_download_url
    if not url or not url.startswith('https://'):
        raise RuntimeError('Kvasir-SEG registry has no trusted HTTPS download URL.')

    archive = download_dir / 'kvasir-seg.zip'
    print('[DATA] Kaggle non-interactive mode: downloading from the public '
          'Simula archive instead of using kagglehub.dataset_download().', flush=True)
    print('[DATA] Official source:', url, flush=True)
    try:
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            # Bound network I/O to the session deadline. TLS verification stays on.
            code = ("from pathlib import Path; "
                    "from scripts.prepare_kvasir_seg import _maybe_download; "
                    "_maybe_download("+repr(url)+", Path("+repr(str(archive))+")")
            run_cmd([sys.executable, '-c', code], cwd=REPO,
                    name='download_kvasir_official', max_seconds=1200)
        result = _find_valid_kvasir_input(download_dir)
        if not result:
            raise RuntimeError('Official ZIP did not have exactly 1000 paired images/masks')
        print('[DATA] Verified 1000 Kvasir-SEG image/mask pairs:', result[1], flush=True)
        return result, 'official_simula_https'
    except Exception as exc:
        failures.append('Simula direct download: '+str(exc))
        print('[DATA WARNING] Official download failed:', str(exc), flush=True)
        archive.unlink(missing_ok=True)

    # Unlike kagglehub, Kaggle CLI can download a public dataset via authenticated
    # API access rather than asking the Notebook service to attach a new input.
    # If Kaggle CLI credentials are unavailable, the clear error below instructs
    # users to attach a dataset in the notebook editor instead.
    try:
        cli_dir = download_dir / 'kaggle_cli'
        cli_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([sys.executable, '-m', 'kaggle', 'datasets', 'download',
                 '-d', 'debeshjha1/kvasirseg', '-p', str(cli_dir)],
                cwd=REPO, name='download_kvasir_kaggle_cli', max_seconds=900)
        result = _find_valid_kvasir_input(cli_dir)
        if not result:
            raise RuntimeError('CLI archive did not contain 1000 Kvasir-SEG pairs')
        print('[DATA] Verified 1000 Kvasir-SEG image/mask pairs:', result[1], flush=True)
        return result, 'kaggle_cli_public_dataset'
    except Exception as exc:
        failures.append('Kaggle CLI download: '+str(exc))
        print('[DATA WARNING] Optional Kaggle CLI fallback failed:', str(exc), flush=True)

    raise RuntimeError(
        'Automatic Kvasir-SEG retrieval failed. In the Kaggle Notebook editor, '
        'click Add Input -> Datasets and attach debeshjha1/kvasirseg BEFORE running '
        'the notebook. Alternatively set DATA_FOLDERS={"kvasir_seg": '
        '"/kaggle/input/<dataset-folder>"} or DATA_ZIPS for a valid ZIP. '
        'Keep FIRST_EXPERIMENT unchanged. Details: ' + ' | '.join(failures)
    )


def preflight_auto_datasets():
    """Fail fast on inaccessible public ZIP hosts, before starting GPU work.

    A 4-byte HTTP probe does not establish download success. Extraction failures
    are handled by the job loop and also recorded as SKIPPED, never COMPLETE.
    """
    if SCOPE != 'AUTO':
        return
    report = BASE/'BT_UNET_DATASET_PREFLIGHT_AUTO.json'
    run_cmd([sys.executable, 'scripts/kaggle/dataset_preflight.py',
             '--output', str(report), '--seconds', '115'], cwd=REPO,
            max_seconds=155, name='auto_dataset_preflight')
    record=json.loads(report.read_text(encoding='utf-8'))
    for dataset in sorted(AUTO_DATASETS):
        details=record.get('datasets',{}).get(dataset, {})
        if details.get('available') and details.get('selected',{}).get('url'):
            AUTO_SOURCE_URLS[dataset]=details['selected']['url']
        else:
            AUTO_UNAVAILABLE[dataset]='preflight failed: '+ ' | '.join(details.get('errors',[]))
    print('[AUTO PREFLIGHT] usable',sorted(AUTO_SOURCE_URLS),
          '| skipped datasets',sorted(AUTO_UNAVAILABLE),flush=True)


def _auto_needs(spec):
    return set(spec.get('sources',[]) + spec.get('targets',[]) + [spec['dataset']])


def _auto_skip(status,index,spec):
    unavailable=sorted(_auto_needs(spec).intersection(AUTO_UNAVAILABLE))
    if not unavailable:
        raise ValueError('Called auto skip for an available dataset')
    skip={'number':index+1,'id':spec['id'], 'dataset':spec['dataset'],
          'unavailable_datasets':unavailable,
          'reason':{name:AUTO_UNAVAILABLE[name] for name in unavailable}}
    status.setdefault('skipped_jobs',[]).append(skip)
    status['next_index']=index+1
    status['stop_reason']='skipped_unavailable_datasets'
    print(f'[AUTO SKIPPED {index+1}/{len(QUEUE)}] {spec["id"]} | '
          + ', '.join(unavailable) + ' | NOT counted as completed',flush=True)
    _save_queue(status)


def setup_datasets():
    sys.path.insert(0,str(REPO))
    from scripts.revision.audit_splits import audit
    datasets=set(SPEC.get('sources',[])+SPEC.get('targets',[])+[SPEC['dataset']])
    if SCOPE == 'AUTO':
        unsupported = datasets - AUTO_DATASETS
        if unsupported:
            raise RuntimeError('AUTO queue unexpectedly includes non-direct datasets: '+str(sorted(unsupported)))
    for dataset in sorted(datasets):
        if dataset in PREPARED:
            if dataset in DATASET_META:
                for key, value in DATASET_META[dataset].items():
                    RUN_META.setdefault(key,{})[dataset]=__import__('copy').deepcopy(value)
            continue
        args=[sys.executable,'scripts/prepare_dataset.py','--dataset',dataset,'--data-root',str(DATA),'--image-size',str(SPEC['image_size'])]
        source_path=os.getenv('BT_DATA_'+dataset.upper()+'_SOURCE')
        zip_path=os.getenv('BT_DATA_'+dataset.upper()+'_ZIP')
        if SCOPE == 'AUTO':
            # Let the existing preparation utility fetch the registered public HTTPS
            # archive directly. No manual Kaggle Input, kagglehub attach, gdown,
            # 1,000-pair audit, or slow content hashing. Its minimum parsing and
            # train/val/test split creation are still required to actually train.
            source_path = None
            zip_path = None
            from src.datasets.registry import get_dataset_spec
            source_url = AUTO_SOURCE_URLS.get(dataset, get_dataset_spec(dataset).default_download_url)
            if not source_url:
                raise RuntimeError('No selected public HTTPS archive URL for '+dataset)
            # Use the exact URL that passed the lightweight preflight, including
            # the official institution-operated mirror when selected.
            args += ['--download-url', source_url]
            RUN_META.setdefault('dataset_provenance',{})[dataset] = {
                'source':'public_https_direct_auto', 'url':source_url,
                'probe_only_not_full_download_verification':True}
            print('[AUTO DOWNLOAD]',dataset,source_url,flush=True)
        if source_path:
            args+=['--source-dir',source_path]
            RUN_META.setdefault('dataset_provenance',{})[dataset]={'source':'explicit_source_dir','path':source_path}
        elif zip_path:
            args+=['--zip-path',zip_path]
            RUN_META.setdefault('dataset_provenance',{})[dataset]={'source':'explicit_zip','path':zip_path}
        elif dataset=='kvasir_seg' and SCOPE != 'AUTO':
            (flag,path,sha),where=_resolve_kvasir_kaggle()
            args += [flag,path]
            RUN_META.setdefault('dataset_provenance',{})[dataset]={
                'source':where,'path':path,'sha256_of_input':sha,
                'original_image_mask_pairs':1000}
        # AUTO mode only: use standard verified HTTPS first. If Simula's CA chain
        # is unavailable on this Kaggle image, allow a visible host-scoped retry.
        download_env=None
        if SCOPE == 'AUTO':
            download_env=os.environ.copy()
            download_env['BT_AUTO_DATASET_TLS_FALLBACK']='1'
        run_cmd(args,cwd=REPO,max_seconds=900 if SCOPE=='AUTO' else 3600,name='prepare_'+dataset,env=download_env)
        if dataset=='kvasir_seg' and SCOPE != 'AUTO':
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
        if SCOPE == 'AUTO':
            # Fast track: no per-file hashes, size assertions, duplication audits,
            # or manual data-source verification. Basic dataset preparation and
            # split generation above remain mandatory to prevent bogus results.
            PREPARED.add(dataset)
            DATASET_META[dataset] = {'dataset_provenance':RUN_META['dataset_provenance'][dataset],
                                     'split_protocol':RUN_META['split_protocol'][dataset]}
            print('[AUTO READY]',dataset,flush=True)
            continue
        # Strict manifest existence + overlap checks, no implicit fallback.
        for part in ['train','val','test']:
            p=DATA/'splits'/dataset/(part+'.txt')
            if not p.is_file() or not p.read_text().strip():raise RuntimeError('Missing nonempty dataset-specific split: '+str(p))
        # Content fingerprints: a fresh session downloads data again; detect source
        # drift instead of quietly claiming a fully identical experiment.
        from src.datasets import build_dataset
        digest=hashlib.sha256()
        unique_files={}
        for split_name in ('train','val','test'):
            for sample_id, image_path, mask_path in build_dataset(
                    dataset,str(DATA),split=split_name,image_size=SPEC['image_size']).samples:
                for tag, file_path in (('I',image_path),('M',mask_path)):
                    unique_files[(tag,str(file_path))]=(tag,sample_id,Path(file_path))
        for (tag,_path),(kind,sid,source) in sorted(unique_files.items()):
            digest.update((kind+'\0'+sid+'\0').encode('utf-8'))
            with source.open('rb') as fh:
                for chunk in iter(lambda:fh.read(1048576),b''):
                    digest.update(chunk)
        RUN_META.setdefault('dataset_content_sha256',{})[dataset]=digest.hexdigest()
        RUN_META.setdefault('dataset_sample_counts',{})[dataset]=len(unique_files)//2
        PREPARED.add(dataset)
        DATASET_META[dataset]={key:__import__('copy').deepcopy(RUN_META.get(key,{}).get(dataset))
                            for key in ('dataset_provenance','split_protocol',
                                        'dataset_content_sha256','dataset_sample_counts')}
    if SCOPE == 'AUTO':
        return
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
        arg=Namespace(groups=SPEC['group'],datasets=SPEC['dataset'],seeds=str(SPEC['seed']),epochs=SPEC['epochs'],data_root=str(DATA),output_root=str(RESULTS/'jobs'),image_size=SPEC['image_size'],batch_size=SPEC['batch_size'],eval_batch_size=4,num_workers=SPEC['num_workers'],device='cuda:0' if not os.environ.get('BT_KAGGLE_TEST_CPU') else 'cpu',split='test',external_configs='',save_predictions=SCOPE!='AUTO')
        options=build_plan(arg)
        selected=[j for j in options if j['variant']==SPEC['group']+'_'+SPEC['variant']]
        if len(selected)!=1:raise RuntimeError('Variant not uniquely found in revision run plan')
        job=selected[0]
        config_path=Path(job['config_path']);config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(yaml.safe_dump(job['config'],sort_keys=False),encoding='utf-8')
        return job['train_command'],Path(job['checkpoint']),[(SPEC['dataset'],job['eval_command'])],job['config'],config_path
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
        cmd=[sys.executable,'scripts/revision/evaluate_revision.py','--checkpoint',str(ckpt),'--dataset',dataset,'--data-root',str(DATA),'--split','test','--batch-size','4','--image-size',str(SPEC['image_size']),'--num-workers',str(SPEC['num_workers']),'--device','cpu' if os.environ.get('BT_KAGGLE_TEST_CPU') else 'cuda:0','--output-dir',str(dst)]
        if SCOPE!='AUTO':cmd.append('--save-predictions')
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
    expected_seconds=cost*SPEC['epochs']*1.08
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
QUEUE_B64_BY_SCOPE = {'CORE': 'eNrdnOtu2zgQhd9Fv42NrcS7QF6lKAhGom2isihQtJt0se++lJz6kjrijESKw/xrE170HZwzpmiG3/7NXrgpdqyVv0T2/PciK7nhrTDZc/bjyFupWSu22SITjSp2bfa8Xi6yrVaHxjbY8MIoLXllfy9L+wMtjrKVqmbs0pexczPGSt5oxvdNJc2hFEzV1Rtj7erx8Z9uiD3fivcHeVzni2yvSlHZYe/0ss3rw579VPqH0PaxbOtGtWaniux5w6tWLLJWCPtM3diLzLw14ur5bO8jt09Um09G/29BQZd8mT+F0qUbO1VdnvJQqjzliWgiS1Ebad5Y24jCaPvzQtVGqwoap08H8J2szyciJxwob9OFA0YvIeEAgZwuGyibxETby1dZb1Gfc1ddfEfxemgCcoDiNkYOYMCIyQEI0RgxQLGJLkWt2LZSL/b/l09XYGDudPUdnHtTEJIJFKQpMgEDRVwmQMCmiAQKGkGJih2va1F1NQAfuavO4UJ3PQkpsZDBGyMWOnpkxULFb4xUyAASEqrZ2anw4eu7hYvdaXgi0iCjhpMGHTJS0qCChRMGGamIspzEwLxoXXr4jtDVyPG1AAVnhBbAzJDSApCUEUqAQhJbh0MtDDQZXVvfmejHjEkOygGKHJiA6OQA17u4jT7g7O4BWotWlgcX8+9WjPGylEYeBePLfA2w+k37O/BQe9+OE5fZZXIYs9vYlJiH7Q0jdnmaCO96ifP1eunH13acuMwIXw8wo3wdnRns6wFihK8j8q6WOF/b9l583Y0Tlxnu6yFmjK/jM0N9PUQM93VE3l9CK3TRPnfy4fDLYAQkAHvdLQHC8KQkAFrfLQDY/7HwS7nZCC3qArxs+dBjgv0/jhSb3PnmCSQHvG8SI3e8eQK5nS+ckam1KEBbK33DSb7uB4iE53bxIB7EvDHxXFYdhHM7NAJa23DT7QXt5avQAH/etJ9g09tx4jK7TAtjdnuXEvOwk2HELkNP5bWr8gtwoeqj0NuugA8zXzVMe1fbM35qW9ue8RPa3/ZAboPfTd40AN+f23r0/WXM6Pgu3+Px3b4nhD/sezy8y/iR0cGu9255EtQus3t3OgnqYY97NnhsYqC3PTs7Ki/E1Z49HZXX7Wevbr7Hah9rI7cnEPuP9mHDpX44r2r+euP77t3jIklxLFhRyVoW5ctHUXr81vC65Lq0yFdNZ12i/34EOsCBF+X0gMMtw/Gs56WKR9YZF970gAMvtekBB1xc42G9k86ylqbEGXL1TIkz1Hp5HKNnwqDrYyqE4VbEVAjDrIFRdP1qZGc76oZrvhfGTvPw+3IH1ha8EmzJ1n+y933Y3ZeExz/xP7a+XGhxmQNx/8mlE+zrmYQlgV59ApBkfAZoSQK79QQgSMjQrGaQYzUmNKtooZlPEmRoVtFCM58kqNCsZg1NqVWjDsZGdRlIjKsZoIG56jJrXOYWAxQViBghgjK3GICQQKTwHxHx2vC631hdBZPiZg5oTG46zRqUGJKAwgKTJERcYkgCiAxMkJChyWeQIx8TmjxaaOaTBBmaPFpo5pMEFZp81tCc/jw7b2QgKc7j424nsB1mDcqcMiAuJhiSIUQ45pQBfCfBkAihAmGfcMerTVAZ3ufABeO9U4RwzCsJIiQuScIFZV5JwIFxCeI/NPf3y72o4OEYRbhkhOeecpoinP3Dc48/VOHhK7wBxk//iqGnK7RqWxbcxK066ELYn3+7fZ6bL6i+WwG43grTt+t/pSpVn766MrJ9b/+4XPZNT1r1APGF8uX6Ly+Uj5iQEWngaAtaJF+HlkjGLpRQUw47kYxdKKHGH5IiGDtvCk0/XEUybf71GX0oi2TI/Osz8jAX0Wx5Umb8ITCyqfKnzMjDY2Tz5E+ZUYfOCCZpzAp6o3lhuj31pjDX5/Qq9bPr/eESSdu4v6gp4LbD+8TUqT1vOiRC7XHLAU08C6y/T5EUMD18JKSAObG+x4yoG/PrFWMs81coxVjmhAsxBDX5MoyATLcIIyCTLcH52kXZXTT9tUownjn9EoxnTrYEw1ATL8EoyFRLMAoy2RK8dn7QrL/cKhjPnH4JxjMnW4JhqImXYBRkqiUYBem5BH//Hyg+jnE=', 'OPTIONAL': 'eNrV2uFu2jAUBeB3yW+00kA3iVeZpsgkDkSFJIrTVmzauy+oDFFNi69976XH/6rWmH7SqXRyyvdf2daM5b5wzU+bbb4usrJr62aXbS5fuIfaNMNDZfqheGnt+OVkjodskVVmNM6O53OvZbFaLqfv2b4r9y7bPC0XWVNNP3KjaSszVEVxOVQU14uKwj2uVt+mlzVHs7OXX2D1lC+yY1fZw/Ty69npUPtyLN664dkO0xtMZ5y10zucb1hk46m3N++W/V58pipf5mue6nwDmmqd80zrnCEarDtf3vdc0fUiQvquZ7XSp6Lypc+vYqZPRTWfPr+JlT4ZDjl3uqETxvjipps1Ycx8yrQjJsEgxkszXEIMSrA0YyXE8EdKJ1CeDlB2h66ttn7L5WBavY2nQ+9vPB1yjwuXpdTneDr0XsfTwfa7cFYCPY+Bwu17DBRi74vjwPa/aA5mD4zmAPZBOzZuHnI+kU4DjPEgd74YD2rLo1pS6XUxHuQmF+OB7G5UCHhbC2Zg9rNgBlojCwFAdrBAAF7rCgQA9qzGbZt8+ZjPK/6eSqdvcVzIvYvjQu1foaZUehjHhdzHOC7IXhYKAu9n0RzMnhbNQetrMRDI3hYJwetvkRDAHvf8alwzFE3rxuHlaNtxnvTP8XSanawUuevJSlHbH1+ZSh+UlSI3RFkpZGfkE8FbpCAQs1cKAtGapgwNsnuK0fDaqBjts/qpKZ3gP6ffbyOE8P2gVgyVUL74eVDMACqh5oPnIfGiN47TX0DTtaKfUfx4KyWKH16gFsn7YL0RpWG5Ub0P1hNdGpUV4dIMRlB4uY4Q2stJrbSquXz59LmYwVRzzUfRp+JlsBYl1dQE1qoB1EF541erpk8H5clerRe9/XT6/Dscnd1JsW7vJMTw9rhWGO/A9AWTxGTG8w7M+aiSkLzASlboPfFZZ6/5qKMi8qZR8zlHReQJnt5DTt8dTn3Rv4qJrhcSknc9q5U+VZ0vhX4dM4mquvlE+m28VIrW4J761NKrPrQoobwxVH1iUUJ50qf4uPK/GYv6uUzwiTGYgTkqBjNwZ0QyJc3hMJqXxlQYzQMeB6km/DkwXII6AIZLACc/MgJ75AtmYM56wQzUIY8KSW66i4Thj3WRMNB5jqwBHuRCDYgTXKgBcXSjGlKZ2WI8yMNajAdySiNDsMezYAbmXBbMwBvI+B81BJ/OBIGYo5ogEHduE0CmOcQpwNOY6BTgwOMdX4s/60kaUQc/SSPgFCjAwx4JBYGY86EgEHVY5BOTmxzFyfhjpDgZdKYUcAIPmHI6xGlTToc4evJ1qcyhslLkoVRWCjmhChCxx1VBIObsKgiUH2R//AEDzX54'}
QUEUE_SHA256_BY_SCOPE = {'CORE': '18988d518b31755ce98920625ca60183b55251881eb94f376fd12976506e24ef', 'OPTIONAL': '24c37d4f27a09e4df6e889f60551a73f6141ed3923540d3f0c281b8f05097ad2'}
_decoded_queues = {key: json.loads(zlib.decompress(base64.b64decode(value)))
                   for key, value in QUEUE_B64_BY_SCOPE.items()}
for _name, _jobs in _decoded_queues.items():
    _digest = hashlib.sha256(json.dumps(_jobs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert _digest == QUEUE_SHA256_BY_SCOPE[_name], f'{_name} source plan checksum mismatch'
AUTO_DATASETS = frozenset({'kvasir_seg', 'kvasir_instrument', 'isbi2012', 'hyper_kvasir_seg', 'bus_uclm'})
if SCOPE in ('ALL','AUTO'):
    # Keep original job specs completely unchanged. Only the presentation order is updated:
    # primary comparisons, reviewer ablations, convergence, then extensions and baselines.
    _joined = _decoded_queues['CORE'] + _decoded_queues['OPTIONAL']
    def _priority(spec):
        typ, group = spec['type'], spec.get('group', '')
        group_rank = {'convergence':0,'factorial':1,'residual':2}
        if typ == 'revision': rank=group_rank.get(group,3)
        elif typ == 'standard': rank=4
        elif typ == 'cross': rank=5
        elif typ == 'lowdata': rank=6
        else: rank=7
        dataset_rank={'kvasir_seg':0, 'cvc_clinicdb':1,'cvc_colondb':2,
                      'etis':3,'kvasir_instrument':4,'cvc_300':5,
                      'bus_uclm':6,'isbi2012':7,'hyper_kvasir_seg':8}
        model_rank={'unet':0,'dapr_unet':1}
        return (rank, dataset_rank.get(spec['dataset'],99),
                model_rank.get(spec.get('model',''),10),
                spec.get('model',''), spec.get('variant',''), spec['seed'], spec['id'])
    if SCOPE == 'AUTO':
        # Independent ultrasound experiments, following the existing ISBI standard
        # 2D comparison protocol; source: public Mendeley BUS-UCLM ZIP.
        # Keep the exact archive URL in provenance (the registry's URL ends -1.zip).
        # Do NOT infer the archive is identical to the published V3 release.
        # They are NEW jobs, not renames/substitutions of any polyp dataset.
        import copy
        bus_jobs=[]
        for base_job in _joined:
            if base_job['dataset']=='isbi2012' and base_job['type']=='standard':
                new_spec=copy.deepcopy(base_job)
                new_spec['dataset']='bus_uclm'
                new_spec['id']=base_job['id'].replace('isbi2012','bus_uclm')
                bus_jobs.append(new_spec)
        _joined += bus_jobs
        # Omit jobs needing Google Drive / manual input, including cross-domain
        # jobs whose training or testing target is not directly downloadable.
        _joined = [spec for spec in _joined if
            {spec['dataset'], *spec.get('sources',[]), *spec.get('targets',[])}.issubset(AUTO_DATASETS)]
    QUEUE = sorted(_joined, key=_priority)
    QUEUE_SHA256 = hashlib.sha256(json.dumps(QUEUE,sort_keys=True,separators=(',',':')).encode()).hexdigest()
else:
    QUEUE = _decoded_queues[SCOPE]
    QUEUE_SHA256 = QUEUE_SHA256_BY_SCOPE[SCOPE]
assert len(QUEUE) == {'CORE':129,'OPTIONAL':132,'ALL':261,'AUTO':165}[SCOPE]
assert len({x['id'] for x in QUEUE}) == len(QUEUE)
assert all(x['epochs'] in (30,50,100) and x['batch_size']==6 and x['image_size']==352 for x in QUEUE)

PREPARED = set()
DATASET_META = {}
# URL probes run on Kaggle itself, not on the offline packaging machine.
# Keep skipped job identifiers separate from successfully completed ones.
AUTO_SOURCE_URLS = {}
AUTO_UNAVAILABLE = {}

CROSS_CHECKED = set()
CALIBRATION_CACHE = {}
JOB_START = START
EVAL_AND_OUTPUT_RESERVE = 1800
SESSION = BASE / ('BT_UNET_MULTI_' + SCOPE + '_from_' + str(START_INDEX+1 if args.one_based else START_INDEX))
SESSION_LOG = SESSION/'session_status.json'
SESSION_REPO = REPO_ROOT
SESSION_DATA = SESSION/'data'
SESSION_RESULTS = SESSION/'job_outputs'
COMPACT_RESULTS = SESSION/'compact_results'
SESSION_LOGS = SESSION/'session_logs'
PLAN_CSV = BASE/('BT_UNET_PLAN_'+SCOPE+'.csv')
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
    EVAL_AND_OUTPUT_RESERVE=(240 + 120*len(evaluations) if SCOPE=='AUTO' else 600 + 1200*len(evaluations)) + (6800 if SPEC.get('posthoc') else 0)
    key=_calibration_key(config)
    if key in CALIBRATION_CACHE:
        cost=CALIBRATION_CACHE[key]
        RUN_META['calibration_reused']=True
        RUN_META['calibration_seconds']=round(cost,2)
        print('[REUSED TIME ESTIMATE]',round(cost,1),'seconds/epoch for same configuration',flush=True)
    else:
        # calibrate() has a budget check after measuring. On failure, the time observation is still logged.
        calibrate(train, config, ckpt)
        cost=RUN_META['calibration_seconds']
        CALIBRATION_CACHE[key]=cost
    expected=cost*SPEC['epochs']*1.08
    RUN_META['estimated_full_train_seconds']=round(expected,1)
    RUN_META['time_estimate_multiplier']=1.08
    RUN_META['evaluation_and_archiving_reserve_seconds']=EVAL_AND_OUTPUT_RESERVE
    print('[BUDGET]',SPEC['id'],'expected GPU train',round(expected/3600,2),'h',
          'eval/analysis & output reserve',round(EVAL_AND_OUTPUT_RESERVE/3600,2),'h',
          'remaining',round(remaining()/3600,2),'h',flush=True)
    if expected + EVAL_AND_OUTPUT_RESERVE + CLEANUP_RESERVE > remaining():
        raise RuntimeError('PROJECTED_OVER_BUDGET: defer this job to another T4 session; no partial training starts.')
    return expected

def export_plan():
    PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields=['experiment_number','id','type','group','dataset','model','variant',
            'seed','epochs','image_size','batch_size','needs_extra_sources']
    with PLAN_CSV.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for idx, spec in enumerate(QUEUE,1):
            writer.writerow({
                'experiment_number':idx,'id':spec['id'],
                'type':spec['type'],'group':spec.get('group',''),
                'dataset':spec['dataset'],'model':spec['model'],
                'variant':spec.get('variant',''), 'seed':spec['seed'],
                'epochs':spec['epochs'],'image_size':spec['image_size'],
                'batch_size':spec['batch_size'],
                'needs_extra_sources': False if SCOPE=='AUTO' else (spec['dataset']!='kvasir_seg' or bool(spec.get('targets'))),
            })
    print('[PLAN]',len(QUEUE),'jobs',PLAN_CSV,'sha256',QUEUE_SHA256,flush=True)


def bundle_session(status):
    """Archive only verified completed jobs; failures never advance the counter."""
    if not args.bundle_session:
        return
    archives=[Path(s) for s in status.get('session_archives',[])]
    if SCOPE == 'AUTO' and not archives and not status.get('skipped_jobs'):
        print('[AUTO] No completed experiments or skipped datasets: no empty session ZIP created.', flush=True)
        return
    compact_auto = (SCOPE == 'AUTO' and any(p.is_dir() for p in archives))
    for a in archives:
        if compact_auto:
            if not a.is_dir() or not (a/'paper_row.json').is_file():
                raise RuntimeError('Missing compact completed result directory: '+str(a))
        elif not a.is_file() or not zipfile.is_zipfile(a):
            raise RuntimeError('Cannot bundle missing/invalid completed archive: '+str(a))
    first=START_INDEX+1
    last=status['next_index']
    target=BASE/f'BT_UNET_SESSION_{SCOPE}_{first:04d}_TO_{last:04d}.zip'
    if target.exists():
        raise FileExistsError('Session bundle already exists; not overwriting: '+str(target))
    # A second copy is needed while making a single ZIP. Keep the per-job ZIPs
    # if disk is insufficient, because they contain the verified research outputs.
    bundle_bytes=(sum(p.stat().st_size for a in archives for p in a.rglob('*') if p.is_file())
                  if compact_auto else sum(a.stat().st_size for a in archives))
    if shutil.disk_usage(BASE).free < bundle_bytes + 150*1024*1024:
        print('[BUNDLE SKIPPED: LOW DISK] Individual job ZIPs are preserved.',flush=True)
        return
    temp=target.with_suffix('.zip.partial')
    manifest={
        'scope':SCOPE,'plan_sha256':QUEUE_SHA256,'first_experiment_number':first,
        'last_completed_experiment_number':max(status.get('completed_numbers',[]) or [0]),
        'last_processed_experiment_number':last,'next_experiment_number':last+1,
        'total_experiments':len(QUEUE),
        'git_commit':status.get('git_commit'), 'completed_ids':status['completed_ids'],
        'completed_experiment_numbers':status.get('completed_numbers',[]),
        'skipped_jobs':status.get('skipped_jobs',[]),
        'archives':([{'name':a.name,'format':'compact_metrics','files':len([p for p in a.rglob('*') if p.is_file()])}
                     for a in archives] if compact_auto else
                    [{'name':a.name,'sha256':sha256_path(a),'bytes':a.stat().st_size} for a in archives]),
        'stop_reason':status.get('stop_reason'),
    }
    try:
        with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
            z.writestr('session_bundle_manifest.json',json.dumps(manifest,indent=2))
            for filename in [SESSION_LOG, PLAN_CSV, BASE/('BT_UNET_NEXT_'+SCOPE+'.json'),
                             BASE/('BT_UNET_NEXT_EXPERIMENT_'+SCOPE+'.txt'),
                             BASE/('BT_UNET_SKIPPED_'+SCOPE+'.json'),
                             BASE/('BT_UNET_DATASET_PREFLIGHT_'+SCOPE+'.json')]:
                if filename.exists():
                    z.write(filename,arcname=filename.name,compress_type=zipfile.ZIP_DEFLATED)
            if compact_auto:
                for filename in sorted(BASE.glob('BT_UNET_*RESULTS_AUTO.*')) + [BASE/'BT_UNET_PAPER_VALUES_AUTO.tex']:
                    if filename.is_file():
                        z.write(filename, arcname=filename.name, compress_type=zipfile.ZIP_DEFLATED)
                for archive in archives:
                    for filename in sorted(archive.rglob('*')):
                        if filename.is_file():
                            z.write(filename, arcname=str(Path('completed_jobs')/archive.name/filename.relative_to(archive)),
                                    compress_type=zipfile.ZIP_DEFLATED)
            else:
                for archive in archives:
                    z.write(archive,arcname='completed_jobs/'+archive.name,
                            compress_type=zipfile.ZIP_STORED)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise RuntimeError('Invalid session bundle CRC')
            if compact_auto:
                if len([n for n in z.namelist() if n.endswith('/paper_row.json')]) != len(archives):
                    raise RuntimeError('Incomplete compact result bundle')
            elif len([n for n in z.namelist() if n.startswith('completed_jobs/')])!=len(archives):
                raise RuntimeError('Incomplete session bundle')
        temp.replace(target)
        print('[SESSION ZIP]',target,'size_MB',round(target.stat().st_size/1e6,1),flush=True)
        if not args.keep_job_zips and not compact_auto:
            # The verified session archive contains byte-identical nested job ZIPs.
            # Avoid saving all those files twice in Kaggle's limited Output storage.
            for archive in archives:
                archive.unlink()
            print('[ARCHIVE STORAGE] Individual ZIPs are inside the session ZIP; '
                  'removed duplicate loose copies.',flush=True)
    finally:
        if temp.exists(): temp.unlink()


def _save_queue(status):
    # Backward-compatible with older callers that supplied only a contiguous
    # completed_ids list. Production AUTO runs always supply explicit numbers.
    if 'completed_numbers' not in status:
        first_number=START_INDEX+1
        n=status['next_index']-START_INDEX
        status['completed_numbers']=(list(range(first_number,status['next_index']+1))
                                     if n==len(status.get('completed_ids',[])) else [])
    SESSION_LOG.parent.mkdir(parents=True,exist_ok=True)
    status['session_elapsed_seconds']=round(time.monotonic()-START,1)
    status['remaining_seconds']=round(remaining(),1)
    status['next_start_index']=status['next_index']  # legacy zero-based
    status['next_experiment_number']=status['next_index']+1  # user-facing one-based
    status['last_completed_experiment_number']=max(status.get('completed_numbers',[]) or [0])
    status['last_processed_experiment_number']=status['next_index']
    status['plan_sha256']=QUEUE_SHA256
    tmp=SESSION_LOG.with_suffix('.tmp')
    tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2,default=str)+'\n')
    tmp.replace(SESSION_LOG)
    (BASE/('BT_UNET_NEXT_'+SCOPE+'.json')).write_text(json.dumps({
       'scope':SCOPE,'next_start_index':status['next_index'],
       'next_experiment_number':status['next_index']+1,
       'last_completed_experiment_number':max(status.get('completed_numbers',[]) or [0]),
       'last_processed_experiment_number':status['next_index'],
       'successful_count_this_session':len(status.get('completed_ids',[])),
       'skipped_count_this_session':len(status.get('skipped_jobs',[])),
       'total_experiments':len(QUEUE),
       'plan_sha256':QUEUE_SHA256,
       'completed_in_this_session':status['completed_ids'],
       'note':'Use next_experiment_number in the next independent Kaggle session; skipped jobs are NOT completed.'
    },indent=2)+'\n')
    (BASE/('BT_UNET_NEXT_EXPERIMENT_'+SCOPE+'.txt')).write_text(
        str(status['next_index']+1)+'\n',encoding='utf-8')
    if SCOPE=='AUTO':
        (BASE/'BT_UNET_SKIPPED_AUTO.json').write_text(json.dumps(
            {'plan_sha256':QUEUE_SHA256,'skipped_jobs':status.get('skipped_jobs',[])},
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run_multi_session():
    global SOURCE_META, ARCHIVE
    if not 0 <= START_INDEX <= len(QUEUE):
        raise ValueError('START_INDEX outside job queue')
    export_plan()
    if PLAN_ONLY or args.plan_csv:
        print('PLAN ONLY',SCOPE,'jobs',len(QUEUE),'starting experiment',START_INDEX+1)
        for j,s in enumerate(QUEUE[START_INDEX:],START_INDEX):
            print(f'{j+1:03d}/{len(QUEUE)} {s["id"]} | {s["epochs"]} epochs | {s["model"]} | {s["dataset"]}')
        return
    if START_INDEX == len(QUEUE):
        print('[COMPLETE QUEUE] All planned jobs already indexed; nothing to train.')
        status={'scope':SCOPE,'start_index':START_INDEX,'next_index':START_INDEX,
                'total_jobs':len(QUEUE),'completed_ids':[], 'completed_numbers':[],
                'skipped_jobs':[], 'failed_jobs':[],
                'stop_reason':'all_jobs_completed', 'git_commit':None,'session_archives':[]}
        _save_queue(status)
        return
    SESSION.mkdir(parents=True,exist_ok=True)
    status={'scope':SCOPE,'start_index':START_INDEX,'next_index':START_INDEX,
       'total_jobs':len(QUEUE),'completed_ids':[],'completed_numbers':[],
       'skipped_jobs':[],'failed_jobs':[],
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
        preflight_auto_datasets()
    except BaseException as exc:
        status['stop_reason']='shared_setup_failed: '+str(exc)
        _save_queue(status)
        if args.bundle_session:
            try: bundle_session(status)
            except Exception as bundle_exc:
                print('[WARNING] Could not bundle setup failure:',bundle_exc,flush=True)
        raise
    _save_queue(status)
    for index in range(START_INDEX,len(QUEUE)):
        spec=QUEUE[index]
        _activate(spec)
        if SCOPE=='AUTO' and _auto_needs(spec).intersection(AUTO_UNAVAILABLE):
            _auto_skip(status,index,spec)
            continue
        print('\n'+'='*70,flush=True)
        print(f'[EXPERIMENT {index+1}/{len(QUEUE)}] {spec["id"]}',flush=True)
        print(f'[COUNTER] completed_global={index} next_experiment={index+1}',flush=True)
        print('[SESSION ELAPSED]',round((time.monotonic()-START)/3600,2),'h',flush=True)
        try:
            hard_check(extra=1800)
            if os.environ.get('BT_KAGGLE_TEST_EPOCHS'):
                raise ValueError('Per-model epoch override disabled for scientific comparisons')
            if ALLOW_PAPER_ALIGNED_REIMPLEMENTATIONS:
                os.environ['BT_ALLOW_PAPER_ALIGNED']='1'
            if shutil.disk_usage(BASE).free < 5*1024**3:
                raise RuntimeError('DISK_LOW: insufficient disk space to safely train/package next job')
            # Notebook Output is capped independently of free temporary disk. Leave
            # room for a single combined ZIP, logs and packaging overhead.
            stored_bytes=sum(p.stat().st_size for p in BASE.glob('BT_UNET_RESULTS__*.zip'))
            if stored_bytes > 8*1024**3:
                raise RuntimeError('DISK_LOW: completed job ZIPs exceed 8GiB; '
                                   'bundle/download this session before more jobs')
            baseline_gate()
            if SCOPE=='AUTO':
                try:
                    setup_datasets()
                except (RuntimeError,FileNotFoundError,ValueError) as data_error:
                    # This is a dataset acquisition/parsing problem, NOT a
                    # completed experiment. Do not retry the same dead URL 80x.
                    for name in sorted(_auto_needs(spec)-PREPARED):
                        AUTO_UNAVAILABLE[name]='dataset preparation failed: '+str(data_error)
                    _auto_skip(status,index,spec)
                    continue
            else:
                setup_datasets()
            train,ckpt,evals,config,config_path=make_job()
            _preflight(train,config,ckpt,evals)
            RUN_META['status']='training'
            # Explicit post-train evaluation/analysis/packaging time is held back.
            train_limit=max(20,remaining()-CLEANUP_RESERVE-EVAL_AND_OUTPUT_RESERVE)
            actual_train_seconds=run_cmd(train,cwd=REPO,max_seconds=train_limit,name='train_full')
            RUN_META['actual_train_seconds']=round(actual_train_seconds,2)
            if SCOPE=='AUTO':
                # This measured full run is more informative than extrapolation
                # from one cold-start calibration epoch for subsequent seeds.
                key=_calibration_key(config)
                previous=CALIBRATION_CACHE.get(key)
                observed=actual_train_seconds/SPEC['epochs']
                CALIBRATION_CACHE[key]=observed
                print('[LEARNED TIME] measured seconds/epoch',round(observed,2),
                      'next-seed planning cost',round(CALIBRATION_CACHE[key],2),flush=True)
            if not ckpt.is_file():
                raise FileNotFoundError('Best checkpoint missing: '+str(ckpt))
            RUN_META['status']='evaluating'
            for label, command in evals:
                run_cmd(command,cwd=REPO,max_seconds=min(1800,remaining()-CLEANUP_RESERVE-700),name='eval_'+label)
            if SPEC.get('posthoc'):
                postprocess(ckpt)
            for ds in sorted(set(SPEC.get('sources',[])+SPEC.get('targets',[])+[SPEC['dataset']])):
                for part in ('train','val','test'):
                    src=DATA/'splits'/ds/(part+'.txt')
                    if src.is_file():
                        dest=RESULTS/'split_manifests'/ds/(part+'.txt')
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(src,dest)
            RUN_META['status']='complete'
            if SCOPE=='AUTO':
                from scripts.kaggle.results_export import export_completed, update_reports, print_paper_result
                paper_row, compact_dir=export_completed(index+1,SPEC,evals,RESULTS,COMPACT_RESULTS,RUN_META)
                all_completed=update_reports(BASE,COMPACT_RESULTS,QUEUE)
                print_paper_result(paper_row)
                print('[SESSION RESULTS]',len(all_completed),'completed; compact export without weights/images',flush=True)
                archive_path=compact_dir
            else:
                package_all()
                archive_path=ARCHIVE
            # Only verified summaries remain. The next job reuses shared datasets.
            shutil.rmtree(RESULTS,ignore_errors=True)
            status['completed_ids'].append(spec['id'])
            status.setdefault('completed_numbers',[]).append(index+1)
            status['session_archives'].append(str(archive_path))
            status['next_index']=index+1
            status['stop_reason']='more_jobs_remaining'
            _save_queue(status)
            print('[EXPERIMENT COMPLETE]',index+1,'/',len(QUEUE),flush=True)
            print('[NEXT EXPERIMENT NUMBER]',index+2,flush=True)
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
                if SCOPE != 'AUTO':
                    try:
                        ARCHIVE=BASE/('BT_UNET_FAILED__'+spec['id']+'.zip')
                        package_all()
                    except Exception as package_exc:
                        print('[FAILED TO ARCHIVE LOGS]',package_exc)
                else:
                    print('[FAILED JOB] No unverified metrics or heavy checkpoint ZIP exported.',flush=True)
            break
        finally:
            _save_queue(status)
    if status['next_index'] == len(QUEUE):
        status['stop_reason']='queue_processed_with_skips' if status.get('skipped_jobs') else 'all_jobs_completed'
    _save_queue(status)
    if SCOPE == 'AUTO':
        from scripts.kaggle.results_export import update_reports
        update_reports(BASE,COMPACT_RESULTS,QUEUE)
    if args.bundle_session:
        try:
            bundle_session(status)
        except Exception as err:
            print('[SESSION BUNDLE FAILED; per-job ZIPs preserved]',repr(err),flush=True)
    # A failed job may already have finished hours of training and evaluation.
    # Keep its checkpoint and per-image metrics for repair/re-export instead of
    # irreversibly deleting them. Successful sessions retain compact cleanup.
    failed = (status.get('stop_reason') or '').startswith(('blocked_or_failed:', 'shared_setup_failed:'))
    if failed:
        print('[RECOVERY] Failure detected; preserving unarchived scratch, checkpoints, and evaluation summaries.', flush=True)
    if not args.keep_scratch and not failed:
        # Kaggle saves /kaggle/working as notebook output: remove data and temporary
        # full checkpoints after verified job archives were created.
        shutil.rmtree(SESSION_DATA,ignore_errors=True)
        shutil.rmtree(SESSION/'job_work',ignore_errors=True)
        shutil.rmtree(SESSION_RESULTS,ignore_errors=True)
        print('[SCRATCH CLEANUP] Removed transient datasets/job working directories',flush=True)
    print('\n'+'='*75)
    print('SCOPE',SCOPE,'COMPLETED THIS SESSION',len(status['completed_ids']))
    print('LAST SUCCESSFUL EXPERIMENT NUMBER =',max(status.get('completed_numbers',[]) or [0]))
    print('LAST PROCESSED EXPERIMENT NUMBER =',status['next_index'])
    print('AUTO DATASET SKIPPED JOBS =',len(status.get('skipped_jobs',[])))
    print('NEXT_EXPERIMENT =',status['next_index']+1)
    print('LEGACY NEXT_START_INDEX =',status['next_index'])
    print('REMAINING PLANNED JOBS =',len(QUEUE)-status['next_index'])
    print('STATUS FILE:',SESSION_LOG)
    print('NEXT SESSION STATE:',BASE/('BT_UNET_NEXT_'+SCOPE+'.json'))
    print('RESULTS: compact BT_UNET_SESSION_AUTO_*.zip, 165-row BT_UNET_PAPER_RESULTS_AUTO.csv'
          if SCOPE=='AUTO' else 'RESULTS: each BT_UNET_RESULTS__*.zip in /kaggle/working')
    print('STOP REASON:',status['stop_reason'])
    print('='*75,flush=True)
    # A real setup/training failure must not masquerade as a successful Kaggle
    # cell. Budget deferrals are expected and still exit successfully.
    if (status.get('stop_reason') or '').startswith(('blocked_or_failed:', 'shared_setup_failed:')):
        raise SystemExit(1)

if __name__=='__main__':
    run_multi_session()
