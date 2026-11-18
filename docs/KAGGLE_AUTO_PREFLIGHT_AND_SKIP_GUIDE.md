# Kaggle AUTO queue — source preflight, skip manifest, and recovery

## What changed

The fixed AUTO queue is **165 experiments**. The original 153 jobs are unchanged; 12 new exploratory BUS-UCLM ultrasound comparison jobs were added. The original ALL (261), CORE (129), OPTIONAL (132) queues are untouched. No core BT U-Net model code has changed.

Data sources are checked on Kaggle at runtime with a fast, streaming 4-byte ZIP signature probe. This is **not a full download/validation**, does not hash or audit data and cannot guarantee that the complete archive will be downloadable. An HTTPS URL being published on a project page does not mean a Kaggle runtime can download it. A probe that fails marks that dataset unavailable for the current session. The runner skips those dataset jobs and continues to a different dataset. It never says skipped jobs were trained.

| Dataset | Job count | Public archive source | Caveat |
|---|---:|---|---|
| Kvasir-SEG | 81 | `https://datasets.simula.no/downloads/kvasir-seg.zip` | Kaggle previously reported SSL chain failures; host-scoped fallback tries reduced TLS verification only after SSL error |
| Kvasir-Instrument | 36 | `https://datasets.simula.no/downloads/kvasir-instrument.zip` | 590 endoscopic tool image/mask pairs according to publisher |
| ISBI-2012 | 12 | `https://downloads.imagej.net/ISBI-2012-challenge.zip` | repository evaluates subsets of **labeled training slices**, **not** the official challenge test |
| Hyper-Kvasir segmented | 24 | Simula official archive; alternative `https://huggingface.co/datasets/SimulaMet-HOST/HyperKvasir/resolve/main/hyper-kvasir-segmented-images.zip` | possible overlap with Kvasir-SEG; do not describe as an independent external test without an overlap audit |
| BUS-UCLM | 12 | public Mendeley S3 ZIP from repository dataset registry | ultrasound rather than polyp segmentation; the archive URL refers to `-1.zip`, so document this exact version instead of silently claiming V3 equivalence |

The Simula Hyper-Kvasir directory listing may present a leading-underscore archival filename; the repository tries its existing public URL and then the original institution's Hugging Face mirror. **No third-party processed or re-labeled substitute** is silently used.

## How to use

1. Push **this** corrected source into GitHub. Earlier commit `ae20d11...` does not contain this fix.
2. Upload `BT_UNet_KAGGLE_11H_AUTO.ipynb` and select a Kaggle GPU with Internet enabled.
3. In session 1 set `FIRST_EXPERIMENT = 1`, `PINNED_COMMIT = ''`, `SESSION_HOURS = 11.0`.
4. Run the cell. It clones GitHub, installs dependencies, probes 5 sources in Kaggle (no full dataset download), runs available jobs and packages the results.
5. Download `BT_UNET_SESSION_AUTO_####_TO_####.zip` and copy `BT_UNET_NEXT_EXPERIMENT_AUTO.txt` into the next notebook's `FIRST_EXPERIMENT` setting. Keep the same Git SHA pinned for reproducible subsequent sessions.

### Files printed/saved

- `BT_UNET_PLAN_AUTO.csv` — immutable 165-job experiment mapping with numeric IDs.
- `BT_UNET_DATASET_PREFLIGHT_AUTO.json` — URLs tried, SSL verification result, sources available **in that Kaggle session**.
- `BT_UNET_NEXT_EXPERIMENT_AUTO.txt` — **next not-yet-processed experiment number**.
- `BT_UNET_SKIPPED_AUTO.json` — list of experiments that **were not run** due to a data problem.
- `BT_UNET_SESSION_AUTO_####_TO_####.zip` — includes successfully completed job archives and available skip/progress metadata.

When one dataset is inaccessible and a later one is usable, the counter advances across the skipped numbers. **Do not calculate the next number from the number of successful jobs.** Use the NEXT file; the completion and skipped counts are separate. If you later want to retry unavailable datasets, deliberately restart from the first skipped experiment with a separate Kaggle session and retain both its result ZIP and prior skip report.

If no dataset works, preflight still records the reasons, no fabricated result ZIP is generated, and all skipped positions are accounted for. Any non-dataset training error is still a failure; it halts the queue and leaves that experiment unprocessed.

## Merging multiple sessions

```bash
python scripts/kaggle/merge_sessions.py \
  --input-dir kaggle_sessions --output BT_UNET_AVAILABLE_RESULTS.zip --scope AUTO
```

The merge records the number of actual successful experiments and skipped jobs. `--require-complete` **fails** when any jobs were skipped or not processed. Do not report a merged package as results from all 165 experiments unless every one has a valid result ZIP.

## Offline test and live limitation

```bash
python -m pytest -q
python scripts/kaggle/multi_t4_runner.py --scope AUTO --one-based --plan-only
```

The code and skip bookkeeping can be tested offline. This development environment does not have network DNS access to Kaggle, Simula, ImageJ, Mendeley, or Hugging Face, so the exact download throughput and source availability **cannot** be guaranteed until the short preflight executes on Kaggle itself. A successful 4-byte probe still does not guarantee successful full-archive download. Data preparation is capped for AUTO; any failure is explicitly recorded rather than repeatedly attempting the same unavailable dataset.
