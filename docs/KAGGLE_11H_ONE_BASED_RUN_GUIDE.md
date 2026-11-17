# BT U-Net — Kaggle 11-hour runner (one-based experiment numbering)

## What is included

- **261 existing 2D experiments**, preserving original seeds, configurations, image size, batch size and epochs: 129 original `CORE` and 132 original `OPTIONAL` experiments.
- A single deterministic queue reorders them to start with Kvasir-SEG U-Net vs BT convergence controls, then factorial/residual, remaining baselines/cross-dataset/low-data tasks.
- Each experiment trains **independently from scratch** and has a unique configuration, trained checkpoint and per-image evaluation results. The dataset download and preparation cache is **shared inside each Kaggle session**; every fresh Kaggle session clones the GitHub repository and prepares data again.
- Job counter is **1-based**. Experiment 1 is the first job, experiment 261 is the last. After finishing 261 jobs the next number is **262**, meaning fully completed.
- The queue is fixed and checksum-verified. The trained source is pinned by Git SHA; download/split fingerprints are written to job metadata.
- Completed jobs are archived separately, then optionally nested into one session ZIP. Partial or failed jobs **never advance** the counter.

**Not included automatically:** real CT/MRI volumetric experiments, verified external official spectral models, clinical annotation requirements. Those require verified datasets, genuine third-party code/weights and their own reviewed protocols. The 261-job queue is NOT evidence that those separate reviewer requirements were completed.

## 1. Publish the corrected code to GitHub (once)

The updated Kaggle runner **must be committed and pushed** before trying to clone it in Kaggle. The public GitHub URL is `https://github.com/tydeptrai21042004/BT_unet.git`. The notebook intentionally rejects an old runner or incomplete clone rather than silently executing an older method.

From Ubuntu / WSL:

```bash
# Have the updated folder on disk and make a clean checkout of your own repo.
git clone https://github.com/tydeptrai21042004/BT_unet.git
# Copy files from BT_unet-main/ of the FULL corrected ZIP into BT_unet/ (preserve all subfolders).
cd BT_unet
git add .
git commit -m "Add reviewer revision and Kaggle 1-based 11h runner"
git push origin main
```

Do not add downloaded data, checkpoints, ZIP output or private credentials to Git.

## 2. Use the supplied Kaggle notebook

Upload `BT_UNet_KAGGLE_11H.ipynb` to Kaggle. In Kaggle Notebook settings enable **GPU** (e.g. Tesla T4) and **Internet**. If needed, attach original datasets using **Add Input**. In the notebook edit these settings:

```python
FIRST_EXPERIMENT = 1  # first Kaggle session
PINNED_COMMIT = ""    # paste the SHA printed during the first run for all later sessions
SESSION_HOURS = 11.0
DATA_FOLDERS = {}
DATA_ZIPS = {}
REVIEWED_SPLITS_DIR = ""  # use reviewed/patient-grouped or benchmark splits when available
```

The notebook clones **GitHub at the beginning of each Kaggle run**, checks the current SHA, downloads/prepares data and runs:

```bash
python scripts/kaggle/multi_t4_runner.py \
  --scope ALL --one-based --start-index 1 \
  --max-hours 11 --bundle-session
```

The whole runtime budget includes cloning, dependency setup, downloading datasets, calibration, training, evaluation and ZIP creation. The scheduler reserves time for outputs. It deliberately declines to start experiments that are projected to exceed the remaining budget. When Kaggle interrupts unexpectedly, the currently running experiment is incomplete and must be retried.

### Dataset downloads

Kvasir-SEG now follows **attached Kaggle Input → official Simula HTTPS ZIP → authenticated Kaggle CLI fallback**; it never asks `kagglehub` to attach a new dataset inside a non-interactive run. Every accepted Kvasir source is checked for **exactly 1000 image/mask pairs**. Other datasets use URLs from `src/datasets/registry.py`. **Availability of these external sites is not guaranteed.** If one is blocked, attach the dataset to Kaggle Inputs and supply either:

```python
DATA_FOLDERS = {'etis': '/kaggle/input/my-etis-dataset'}
# or
DATA_ZIPS = {'cvc_clinicdb': '/kaggle/input/my-pranet-dataset/TrainDataset.zip'}
```

When an official or patient-level split protocol is needed, provide `REVIEWED_SPLITS_DIR` with `dataset/train.txt`, `dataset/val.txt`, `dataset/test.txt`. Default deterministic image-level splits are not automatically patient-independent or identical to the original benchmark's official protocol.

### Fix for `New Datasets cannot be attached in non-interactive sessions`

This was the original failure before experiment #1 began. The updated runner avoids the blocked `kagglehub.dataset_download()` code path. **Upload/push the corrected repository to GitHub before running the notebook.** `PINNED_COMMIT` must be empty on the first run with the new repository (do not reuse a pre-fix commit SHA).

When using Kaggle's notebook **Save Version / Run All**, the safest fallback is to attach the dataset in the notebook editor *before* running: click **Add Input → Datasets**, search `debeshjha1/kvasirseg` and add it. In most cases its automatically mounted path will be detected; if not, explicitly set `DATA_FOLDERS = {'kvasir_seg': '/kaggle/input/<actual-mounted-name>'}`. The notebook must have Internet enabled to try the public direct HTTPS download.

In the original failed log, `LAST_COMPLETED = 0`, `NEXT_EXPERIMENT = 1`. Keep `FIRST_EXPERIMENT = 1`. The previous empty ZIP `...0001_TO_0000.zip` is a failure log bundle, **not** a completed job. If you rerun the updated notebook in the same Kaggle session, it verifies and refreshes its Git clone without deleting previous result ZIPs.

**Important:** A direct public server might be unavailable from Kaggle, and Kaggle CLI requires valid credentials/permissions. The code cannot force an external provider to serve a file; if both download routes fail, attach the original dataset using Add Input. It never creates synthetic medical data as a fallback.

## 3. Progress and ZIPs after each session

Look in Kaggle **Output** (`/kaggle/working`):

```text
BT_UNET_PLAN_ALL.csv                       # all 261 numbered experiments
BT_UNET_NEXT_EXPERIMENT_ALL.txt             # single integer, e.g. 17
BT_UNET_NEXT_ALL.json                      # progress status and queue checksum
BT_UNET_SESSION_ALL_0001_TO_0016.zip        # single ZIP of completed jobs from the session
BT_UNET_RESULTS__<experiment-id>.zip        # only if session bundle failed or --keep-job-zips
BT_UNET_GITHUB_COMMIT.txt                  # exact cloned source commit
```

In the example, **1–16 are completed** and the next session starts at **17**. For the next run, open the same notebook, set only:

```python
FIRST_EXPERIMENT = 17
```

Keep `REPO_URL`, `PINNED_COMMIT`, datasets and split protocol identical across sessions. The runner **does not reload previous checkpoints**, because each numbered experiment is independent. Individual job ZIPs are nested inside the verified session ZIP; duplicate loose ZIPs are deleted after bundling to conserve Kaggle output space. The prior session ZIPs must still be saved/downloaded by you; Kaggle's new session is not a guaranteed permanent archive of former `/kaggle/working` files.

**If experiment 17 fails:** the counter remains **17**. Never input 18 unless experiment 17 has a verified ZIP. An over-budget experiment with no completed checkpoint cannot be skipped silently. If it cannot fit into any 11-hour session, its training protocol must be reconsidered explicitly rather than misrepresenting an incomplete run as finished.

## 4. Combine session ZIPs after finishing all 261

Place all saved session ZIP files into a local folder `kaggle_sessions` and run:

```bash
python scripts/kaggle/merge_sessions.py \
   --input-dir kaggle_sessions \
   --output BT_UNET_ALL_261_RESULTS.zip \
   --require-complete
```

The merger validates numbered intervals, repository commit and queue checksum, individual archive SHA256 hashes and ZIP CRCs. It refuses missing/overlapping experiment ranges. The final archive contains 261 nested job ZIPs. It may require substantial local disk space.

## 5. Notes on publication and execution limits

- Kaggle currently documents up to **12 hours** of notebook execution for CPU/GPU notebooks. The runner uses at most 11 hours plus a cleanup reserve within this limit; outputs should be saved by **Save Version / Run All** and downloaded after success.
- Kaggle output is limited: the combined session bundle duplicates per-job ZIP storage. If there is insufficient space, the runner keeps individual job ZIPs and reports that the session bundle was not made. Download those ZIPs.
- If Internet is disabled, GitHub cloning and external data downloads cannot work.
- Specialized baseline implementations in this repository are explicitly labeled *paper-aligned local reimplementations*, not author-official models. Do not represent their results as official spectral benchmark evidence.
- The runner estimates epoch costs but cannot guarantee a job finishes; it reserves time for evaluation and packing, and refuses to mark a partial run successful.
