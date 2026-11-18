# BT U-Net — 165 AUTO experiments: compact paper-ready results

## What changed (without changing experiments)

- The exact experiment plan is unchanged (165 rows, five datasets, seeds 42/1337/2024, 30/50/100 epochs). Comparing `BT_UNET_PLAN_AUTO.csv` before and after patch gives the same SHA-256 file hash.
- `evaluate_revision.py` now calculates the **real held-out test loss** from the checkpoint's configured loss (`train.loss`: `bce_dice`, `dice` or `structure`). Its definition is **main segmentation head only**, matching the repository's validation evaluator with `include_aux_loss_in_eval=False`, and is the arithmetic mean of test-batch losses.
- `summary.json`: `test_loss`, `loss_name`, `metrics_mean.loss`; image-level mean/standard deviation/median, and extra metrics: `boundary_f1`, `hd95_pixels`, `assd_pixels`, `pixel_accuracy`, `specificity`, `balanced_accuracy`, `f2_score`. HD95/ASSD are undefined for certain empty-prediction cases; they are excluded from the finite-value means, with counts reported separately.
- AUTO no longer packages `.pt` checkpoints or `.png` predictions. It creates one small session ZIP with evaluation summaries, per-image CSV files, train `summary.json`, `history.csv` and configuration/provenance per completed experiment.
- The scheduler now learns the observed actual seconds per epoch of a completed training job and uses a 1.08 planning factor, rather than treating every single calibration epoch as twice as expensive. This addresses the premature stop observed after experiment #1. The full 100/50/30-epoch settings remain unchanged.

## Running on Kaggle

**You must push the patched project source to `https://github.com/tydeptrai21042004/BT_unet` first.** Otherwise the notebook clones the old scheduler and nothing changes.

1. Open `BT_UNet_KAGGLE_11H_AUTO.ipynb` on Kaggle. Enable Internet and GPU.
2. The previously verified session completed experiment **#1**. Set `FIRST_EXPERIMENT = 2` for the next independent session. If you have pushed changes, use **the new** Git commit SHA for `PINNED_COMMIT` (do not pin the old `44976cf...` SHA).
3. Run the cell. It repeats: prepare reused data -> calibrate/reuse observed speed -> train complete epochs -> evaluate on held-out test -> export/print metrics -> start next experiment if enough budget remains.
4. The runner stops and finalizes output before the 11-hour wall-clock budget. It **cannot guarantee using exactly 11 hours**: if the next *complete* experiment cannot safely fit, it defers that job rather than starting a likely-to-be-killed run.
5. Download `BT_UNET_SESSION_AUTO_XXXX_TO_YYYY.zip` from Kaggle Output, along with optional standalone `BT_UNET_PAPER_RESULTS_AUTO.csv`. Set the next `FIRST_EXPERIMENT` from printed `NEXT SESSION FIRST_EXPERIMENT`.

### Contents of the session ZIP

- `BT_UNET_PAPER_RESULTS_AUTO.csv`: 165 numbered rows; `---` for uncompleted experiments; 13 quantitative fields including the six paper columns.
- `BT_UNET_COMPLETED_RESULTS_AUTO.csv` and `.json`: only completed experiments.
- `BT_UNET_PAPER_VALUES_AUTO.tex`: six metrics for each completed experiment to paste into its matching numbered LaTeX table row.
- `completed_jobs/0001/evaluations/<dataset>/summary.json` and `per_image.csv` (one such folder per completed experiment); `history.csv`, `training_summary.json`, `resolved_train_config.yaml`, `paper_row.json`, `provenance.json`; optional compact post-hoc summary CSV/JSON.
- `session_bundle_manifest.json`, `BT_UNET_PLAN_AUTO.csv` and restart state files.

A completed job is printed as:

```text
[PAPER RESULT #002/165] kvasir_seg unet seed=1337 | DICE=... | IOU=... | PRECISION=... | RECALL=... | MAE=... | LOSS=... | BOUNDARY_F1=... ...
[LATEX SIX CELLS] \(0.xxxxxx\) & \(0.xxxxxx\) & ...
```

## Combining results across all Kaggle sessions

Place all downloaded `BT_UNET_SESSION_AUTO_*.zip` files in `downloads/` and run:

```bash
python scripts/kaggle/merge_paper_metrics.py \
  --input-dir downloads --output-dir paper_results
```

If you include your **old** completed experiment #1 (the original 71-MB session ZIP) together with newly patched sessions, Git commits differ. Verify that the training protocol is unchanged, then explicitly use `--allow-mixed-commits`. The script rejects different experiment manifests or duplicate/mismatched experiment IDs.

For the legacy experiment #1, Dice/IoU/precision/recall/MAE and boundary metrics can be imported from its existing test `summary.json`. Its **test loss is unavailable** and is left as `---`; it can only be recovered by rerunning evaluation with the saved checkpoint and the original test split and images.

### Important dataset limitation

Your uploaded Kaggle log reported BUS-UCLM's current Mendeley ZIP endpoint returning **HTTP 403**. Those 12 runs remain `SKIPPED`, *not* completed, until the actual BUS-UCLM source is accessible. The exporter does not fabricate metrics or replace the dataset. The 165-run manifest remains fixed.

### Precise interpretation of loss

The new `LOSS` column is **held-out test loss**, not training loss or best validation loss. For fair multi-model comparisons, note that its numerical definition depends on `train.loss` in each model configuration; compare Dice and IoU as common task metrics, and compare loss only among models using the same definition.
