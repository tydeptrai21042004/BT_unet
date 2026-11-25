"""Kaggle 12h AUTO queue with automatic same-dataset recovery.

Upload this notebook on Kaggle, enable Internet/GPU, and edit FIRST_EXPERIMENT
between Kaggle sessions. Requires pushing the full updated ZIP into GitHub first.
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
from pathlib import Path

# EDIT ONLY FIRST_EXPERIMENT when starting the next independent Kaggle session.
REPO_URL = 'https://github.com/tydeptrai21042004/BT_unet.git'
FIRST_EXPERIMENT = 1
LAST_EXPERIMENT = 165  # Inclusive; use 12, 93, or 105 for targeted recovery.
PINNED_COMMIT = ''  # Pin the NEW SHA after pushing the updated code; the previous SHA runs old behavior.
SESSION_HOURS = 12.0
TOTAL_EXPERIMENTS = 165

assert 1 <= FIRST_EXPERIMENT <= TOTAL_EXPERIMENTS + 1
assert FIRST_EXPERIMENT <= LAST_EXPERIMENT + 1 <= TOTAL_EXPERIMENTS + 1
assert 0.25 <= SESSION_HOURS <= 12
started = time.monotonic()
work = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
checkout = work/'BT_unet_github_checkout'
next_file = work/'BT_UNET_NEXT_EXPERIMENT_AUTO.txt'
if next_file.exists():
    next_stored = int(next_file.read_text().strip())
    if FIRST_EXPERIMENT < next_stored:
        raise RuntimeError(f'Experiments through #{next_stored-1} already finished in this session. '
                           f'Set FIRST_EXPERIMENT={next_stored} or start a fresh Kaggle session.')
if checkout.exists():
    if not (checkout/'.git').is_dir():
        raise RuntimeError('Checkout directory exists but is not a Git repository: '+str(checkout))
    remote = subprocess.check_output(['git','-C',str(checkout),'remote','get-url','origin'],text=True).strip()
    if remote.rstrip('/').lower() != REPO_URL.rstrip('/').lower():
        raise RuntimeError('Checkout remote differs; refusing to remove non-runner files')
    shutil.rmtree(checkout)
print(f'[AUTO DATASETS] Automatic Kaggle/Hugging Face/HTTPS sources | {TOTAL_EXPERIMENTS} independent experiments',flush=True)
print(f'[START] #{FIRST_EXPERIMENT}/{TOTAL_EXPERIMENTS} | budget {SESSION_HOURS}h',flush=True)
subprocess.run(['git','clone',REPO_URL,str(checkout)],check=True)
if PINNED_COMMIT:
    subprocess.run(['git','-C',str(checkout),'checkout','--detach',PINNED_COMMIT],check=True)
commit = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
(work/'BT_UNET_GITHUB_COMMIT_AUTO.txt').write_text(commit+'\n')
print('[COMMIT]',commit,flush=True)
runner=checkout/'scripts/kaggle/multi_t4_runner.py'
if not runner.is_file():
    raise FileNotFoundError('Kaggle runner missing from GitHub. Push the updated FULL ZIP first.')
source=runner.read_text(encoding='utf-8')
if "'AUTO':165" not in source or 'epoch_recovery_no_cold_start_calibration' not in source:
    raise RuntimeError('GitHub has the old scheduler/exporter. Push the updated FULL ZIP to GitHub before running.')
os.environ['BT_T4_MONOTONIC_START']=str(started)
os.environ['BT_KAGGLE_WORKDIR']=str(work)
if PINNED_COMMIT: os.environ['BT_UNET_COMMIT']=PINNED_COMMIT
cmd=[sys.executable,str(runner),'--scope','AUTO','--one-based',
     '--start-index',str(FIRST_EXPERIMENT),'--max-hours',str(SESSION_HOURS),
     '--bundle-session','--end-experiment',str(LAST_EXPERIMENT)]
completed=subprocess.run(cmd,cwd=str(checkout),check=False)
if next_file.is_file():
    print('[NEXT SESSION FIRST_EXPERIMENT =]',next_file.read_text().strip())
else:
    print('[SETUP FAILED BEFORE COUNTER] Restart with FIRST_EXPERIMENT =',FIRST_EXPERIMENT)
archives=sorted(work.glob('BT_UNET_SESSION_AUTO_*.zip'))
print('[COMPACT SESSION ZIP FILES]',*[str(x) for x in archives],sep='\n  ')
for label in ('BT_UNET_PAPER_RESULTS_AUTO.csv', 'BT_UNET_COMPLETED_RESULTS_AUTO.csv',
              'BT_UNET_COMPLETED_RESULTS_AUTO.json', 'BT_UNET_PAPER_VALUES_AUTO.tex'):
    candidate=work/label
    if candidate.exists():
        print('[PAPER-READY OUTPUT]',candidate, f'{candidate.stat().st_size} bytes')
print('[NOTE] AUTO ZIPs contain evaluation metrics, per-image CSV and training histories; no weights/images.')
print('[RECOVERY CHECKPOINTS — retain or attach as Kaggle input in next session]', *map(str, (work/'BT_UNET_RECOVERY').glob('*.pt')))
print('[SCHEDULER EXIT CODE]',completed.returncode)
report=work/'BT_UNET_DATASET_PREFLIGHT_AUTO.json'
if report.is_file():
    import json
    checked=json.loads(report.read_text()).get('datasets',{})
    print('[DATASET PREFLIGHT]',{name:('NOT PROBED' if item.get('available') is None else 'READY' if item.get('available') else 'UNAVAILABLE')
          for name,item in checked.items()})
skipped=work/'BT_UNET_SKIPPED_AUTO.json'
if skipped.is_file():
    import json
    missing=json.loads(skipped.read_text()).get('skipped_jobs',[])
    print('[EXPERIMENTS SKIPPED, NOT COMPLETED]',len(missing))
    if missing:
        print('[SKIPPED IDS REPORT]',skipped)

if completed.returncode:
    raise RuntimeError(
        f'AUTO runner failed with code {completed.returncode}. Check the printed '
        f'job log. The next experiment number is preserved; do not skip it.'
    )
