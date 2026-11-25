"""Single-cell Kaggle launcher for 1-based BT U-Net independent experiment queue.

Copy this whole file into one Kaggle Notebook Python cell, or upload the included
.ipynb. Update ONLY FIRST_EXPERIMENT between sessions, provided your repository SHA
and data/split protocol stay the same.

Before using, PUSH the corrected repository (containing the patched
scripts/kaggle/multi_t4_runner.py) to GitHub. The notebook always clones GitHub.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

# ======================= EDIT THESE SETTINGS =============================
REPO_URL = "https://github.com/tydeptrai21042004/BT_unet.git"
FIRST_EXPERIMENT = 1            # Next session: paste NEXT_EXPERIMENT from previous output.
PINNED_COMMIT = ""              # Recommended: paste the full SHA printed on session #1.
SESSION_HOURS = 12.0           # Includes git clone, dependencies, data, experiments, ZIP.

# Data source priority: existing attached Input -> official HTTPS download ->
# authenticated Kaggle CLI. Non-interactive kagglehub attachment is NOT used.
# If network downloads are blocked, use Kaggle editor -> Add Input first.
# Keys are dataset IDs from src/datasets/registry.py, values are mounted paths.
# Existing extracted datasets: DATA_FOLDERS. ZIP archives: DATA_ZIPS.
DATA_FOLDERS = {}  # e.g. {'etis': '/kaggle/input/my-etis-dataset'}
DATA_ZIPS = {}     # e.g. {'cvc_clinicdb': '/kaggle/input/my-data/TrainDataset.zip'}

# If patient/video-level or official splits are available as Kaggle Input:
REVIEWED_SPLITS_DIR = ""  # e.g. '/kaggle/input/bt-reviewed-splits/splits'

# False = run existing paper-aligned local implementations, labeled honestly.
# True = stop on specialized baselines instead of presenting them as official.
STRICT_SPECIALIZED_BASELINES = False
# ===================== END USER SETTINGS ================================

start_monotonic = time.monotonic()
work = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
checkout = work / 'BT_unet_github_checkout'
print(f'[BEGIN] Experiment #{FIRST_EXPERIMENT} / 261',flush=True)
print(f'[SESSION BUDGET] {SESSION_HOURS:.2f} hours total',flush=True)
if not isinstance(FIRST_EXPERIMENT,int) or not 1<=FIRST_EXPERIMENT<=262:
    raise ValueError('FIRST_EXPERIMENT must be 1..261, or 262 when everything is already done')
if not 0.25<=SESSION_HOURS<=12.0:
    raise ValueError('SESSION_HOURS must be in [0.25,12]')

# Prevent accidental duplicates when a user reruns the same cell after some
# successful experiments. After a failed first job, the stored counter remains 1.
progress_file = work / 'BT_UNET_NEXT_EXPERIMENT_ALL.txt'
if progress_file.exists():
    prior_next = int(progress_file.read_text(encoding='utf-8').strip())
    if FIRST_EXPERIMENT < prior_next:
        raise RuntimeError(f'This session already completed experiments before #{prior_next}. '
                           f'Set FIRST_EXPERIMENT = {prior_next} to avoid duplicates.')

if checkout.exists():
    # Only remove this runner-owned checkout, after verifying its origin. Never
    # remove datasets, previous session archives, or an arbitrary user folder.
    if not (checkout / '.git').is_dir():
        raise RuntimeError(f'The expected checkout path exists but is not a Git clone: {checkout}')
    existing_remote = subprocess.check_output(
        ['git', '-C', str(checkout), 'remote', 'get-url', 'origin'], text=True).strip()
    if existing_remote.rstrip('/').lower() != REPO_URL.rstrip('/').lower():
        raise RuntimeError('Existing checkout remote does not match REPO_URL; refusing to delete.')
    import shutil
    print('[RETRY] Removing runner-owned checkout to start a clean clone; prior ZIPs are preserved.', flush=True)
    shutil.rmtree(checkout)

os.environ['BT_T4_MONOTONIC_START'] = str(start_monotonic)
os.environ['BT_KAGGLE_WORKDIR'] = str(work)
if PINNED_COMMIT:
    os.environ['BT_UNET_COMMIT']=PINNED_COMMIT
if REVIEWED_SPLITS_DIR:
    if not Path(REVIEWED_SPLITS_DIR).is_dir():
        raise FileNotFoundError('REVIEWED_SPLITS_DIR is missing: '+REVIEWED_SPLITS_DIR)
    os.environ['BT_SPLITS_DIR']=REVIEWED_SPLITS_DIR
for dataset, directory in DATA_FOLDERS.items():
    if not Path(directory).exists(): raise FileNotFoundError(directory)
    os.environ[f'BT_DATA_{dataset.upper()}_SOURCE']=str(directory)
for dataset, archive in DATA_ZIPS.items():
    if not Path(archive).is_file(): raise FileNotFoundError(archive)
    os.environ[f'BT_DATA_{dataset.upper()}_ZIP']=str(archive)

print('[GIT] Cloning',REPO_URL,flush=True)
subprocess.run(['git','clone',REPO_URL,str(checkout)],check=True)
if PINNED_COMMIT:
    # A full clone permits checking out an older, exact revision in a later session.
    subprocess.run(['git','-C',str(checkout),'checkout','--detach',PINNED_COMMIT],check=True)
commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
print('[SOURCE COMMIT]',commit,flush=True)
(work/'BT_UNET_GITHUB_COMMIT.txt').write_text(commit+'\n',encoding='utf-8')

runner=checkout/'scripts/kaggle/multi_t4_runner.py'
if not runner.is_file():
    raise FileNotFoundError('Updated Kaggle runner not found on GitHub. '
                            'Push the corrected ZIP contents to the repository first.')
runner_source = runner.read_text(encoding='utf-8')
if '--one-based' not in runner_source or 'official_simula_https' not in runner_source:
    raise RuntimeError('GitHub does not contain the revised 1-based runner with '
                       'the non-interactive Kaggle dataset-download fix. '
                       'Push the corrected repository and rerun.')

command=[sys.executable,str(runner),
         '--scope','ALL','--one-based','--start-index',str(FIRST_EXPERIMENT),
         '--max-hours',str(SESSION_HOURS),'--bundle-session']
if STRICT_SPECIALIZED_BASELINES:
    command+=['--strict-baselines']
print('[RUN]', ' '.join(command),flush=True)
status=subprocess.run(command,cwd=str(checkout),check=False)
next_file=work/'BT_UNET_NEXT_EXPERIMENT_ALL.txt'
if next_file.exists():
    print('\n[NEXT SESSION] Only change FIRST_EXPERIMENT to:',next_file.read_text().strip())
else:
    print('\n[WARNING] No progress counter: setup may have failed before the queue started.')
print('[ALL RESULTS ZIPs]',*[p.name for p in sorted(work.glob('BT_UNET_SESSION_ALL_*.zip'))],sep='\n - ')
print('[LOOSE JOB ARCHIVES]',len(list(work.glob('BT_UNET_RESULTS__*.zip'))),'(0 is expected when all jobs are in a verified session ZIP)')
if status.returncode:
    raise RuntimeError(f'Experiment scheduler exited with code {status.returncode}; '
                       'check the Kaggle cell log and NEXT_EXPERIMENT file. Do not skip a failed experiment.')
