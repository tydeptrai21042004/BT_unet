# Kaggle AUTO-only dataset queue (11-hour independent sessions)

## Difference from the original ALL queue

`--scope ALL` still contains the original **261 experiments**.
`--scope AUTO` is a separate **153-experiment** queue that contains only experiments
using datasets with public direct HTTPS ZIP URLs registered in `src/datasets/registry.py`:

- `kvasir_seg`: 81 jobs — Simula direct ZIP
- `kvasir_instrument`: 36 jobs — Simula direct ZIP
- `isbi2012`: 12 jobs — ImageJ direct ZIP (evaluation uses the **labeled training volume split**, not official challenge test)
- `hyper_kvasir_seg`: 24 jobs — Simula segmented-images direct ZIP

The queue excludes other datasets with Google Drive links and all cross-dataset jobs
whose target/source includes an excluded dataset. **153 is a new numbering scheme, not
experiments 1–153 of the earlier 261-job plan.** Do not mix `ALL` and `AUTO`
progress counters, CSV manifests or result archives.

## What the AUTO runner does

1. Clone your GitHub repo with the latest corrected code. After the first session,
   set `PINNED_COMMIT` to keep code fixed across sessions.
2. Generate numbered plan `BT_UNET_PLAN_AUTO.csv`.
3. Install requirements, download the required dataset from its registered public HTTPS
   URL using `scripts/prepare_dataset.py`, resize/preprocess it and create reproducible splits.
4. Train the experiment **from scratch**, evaluate the checkpoint and ZIP its outputs.
5. Advance the next counter only after checkpoint, evaluation and ZIP succeed.
6. Stop before 11 hours including setup, evaluation and packaging; combine finished
   jobs into `BT_UNET_SESSION_AUTO_XXXX_TO_YYYY.zip`.

AUTO deliberately **skips optional** data audits: no full-file content hash,
no exact 1,000-pair count check, no KaggleHub dataset attachment and no manual
`Add Input`. Essential ZIP decoding, matching images/masks, sample creation, and
split-generation logic in the dataset preparation module are retained because
training cannot run correctly without them. TLS certificate checking remains enabled.

The datasets are publicly listed, but **network access on Kaggle must still be
available**. A host/network error cannot be bypassed simply by turning off
verification. If a download fails, the runner stops and the next experiment number
remains unchanged; it never reports a missing experiment as complete.

## Quick start

1. Unzip `BT_UNet_Kaggle_AUTO_FULL.zip` and push the repository files to GitHub.
2. Upload `BT_UNet_KAGGLE_11H_AUTO.ipynb` into Kaggle.
3. Enable Internet and a GPU accelerator.
4. Set `FIRST_EXPERIMENT=1` in the notebook (other defaults are ready).
5. Download the `BT_UNET_SESSION_AUTO_*.zip` and `BT_UNET_NEXT_EXPERIMENT_AUTO.txt`
   from Kaggle Outputs after each session.
6. On the next Kaggle session, set `FIRST_EXPERIMENT` to the number in that txt file.

Preview the queue locally with:

```bash
python scripts/kaggle/multi_t4_runner.py --scope AUTO --one-based --start-index 1 --plan-only
```

Merge completed AUTO sessions (only after the last number 153 has finished):

```bash
python scripts/kaggle/merge_sessions.py \
  --scope AUTO --input-dir kaggle_sessions \
  --output BT_UNET_ALL_AUTO_153_RESULTS.zip --require-complete
```

## Research caveats

This queue does not run any previously excluded 2D datasets, volumetric CT/MRI
experiments or external spectral baselines. Per-image splitting for ISBI-2012 is
not a patient-independent official challenge evaluation. Download source availability
has not been live-tested from inside Kaggle. The 11h scheduler can defer a job that
it estimates cannot finish in the remaining session; it does not silently shorten
training or mark an unfinished job as complete.
