# AUTO 324 — revised BT U-Net campaign

This is the active Kaggle queue. The older 165 and 300 queues are preserved for archive compatibility, not for final revised-method results.

## Run

Push the updated repository to GitHub, enable Kaggle Internet and GPU, and open `BT_UNet_KAGGLE_11H_AUTO.ipynb`. In a fresh session start with `FIRST_EXPERIMENT=1`, `LAST_EXPERIMENT=324`, `SESSION_HOURS=12.0`. Attach all prior compact `BT_UNET_SESSION_AUTO_*.zip` archives to `/kaggle/input` and pin the updated Git SHA. On subsequent fresh sessions attach the old and new archives and use the printed next experiment counter. Keep recovery `.pt` files for interrupted jobs. To recover skipped dataset gaps, restart at 1 in a fresh session with all completed archives; existing matching entries are skipped. Do not start at 166 for the revised campaign.

The equivalent plan-only command is `python scripts/kaggle/multi_t4_runner.py --scope AUTO --rerun-proposal --plan-only`. The `--rerun-proposal` flag implicitly enables the dataset expansion and changes the active plan to 324 entries. The ordinary 165/300 flags remain legacy modes.

## Queue

| Rows | Group | Entries |
|---|---|---:|
| 1–165 | Original groups, revised proposal family and unchanged baselines | 165 |
| 166–192 | CVC-ClinicDB | 27 |
| 193–219 | CVC-ColonDB | 27 |
| 220–246 | ETIS | 27 |
| 247–273 | ISIC2018 | 27 |
| 274–288 | LinkNet on five original datasets | 15 |
| 289–300 | Matched Hyper-Kvasir baselines (U-Net, U-Net++, ResUNet++, Attention U-Net) | 12 |
| 301–324 | HSNet and CFA-Net on four added datasets | 24 |
| Total | | 324 |

All 105 proposal-family jobs (93 from the original 165, 12 on the additional datasets) have a distinct `h2__` identity and resolve `model.fourier_hermitian_constraint: true`. This covers the full method, factorial ablations, residual/spatial controls, hyperparameter sensitivity and low-data settings. No topology, seed, epoch budget, optimization setting or baseline identity in rows 1–288 is changed except for the proposal response constraint and proposal IDs. Disabled amplitude/phase variants retain their ablation semantics. Old proposal rows cannot satisfy the new IDs, and their recovery files cannot be selected. The prior 12 isolated H2 runs at rows 289–300 are replaced by matched Hyper-Kvasir controls because H2 is now the active proposal everywhere. The same numerical row number does not establish experimental identity.

LinkNet is scheduled for all nine datasets with three seeds per dataset (27 entries). The four new datasets also have 3 seeds each for HSNet and CFA-Net, which are existing paper-aligned models. The new schedule has 105 proposal-family entries and 219 baseline entries (including LinkNet). There are no new model parameters from the symmetry correction. The correction is not a measured performance gain.

## Reuse and metrics

Only unchanged baseline IDs with compatible manifests, finite core metrics, clean test provenance and matching job specs are reused from the compact archives. Completed H2 jobs can be reused in subsequent sessions. Reuse does not independently verify that older data membership equals newly acquired data; audit the split/content fingerprints and archived manifests before making paired scientific claims. The previous 114 recorded manuscript entries are not a results archive; of those entries, 28 are unchanged baseline entries and 86 belong to the superseded proposal family. If all 28 have valid compact archives and no other runs are already complete, 296 active entries remain. Older nested-format archives are not automatically imported. New metrics absent from old archives remain unavailable unless recoverable from existing per-image records or weights. The 24 segmentation/probability fields and efficiency outputs from AUTO 300 are retained. Seed-42 revised proposal checkpoints, convergence checkpoints, and selected standard checkpoints are saved for future diagnostics.

Use `python scripts/kaggle/merge_paper_metrics.py --input-dir downloaded_sessions --output-dir paper_results --revised-plan` to merge old and new compact sessions into the 324-row paper table. Keep original archives for additional provenance, predictions and statistical analysis. Do not use the legacy strict contiguous `merge_sessions.py` tool to combine different plan generations. This revision does not fabricate any new results.

## Acquisition and limits

All nine datasets retain the automatic acquisition/fallback paths from AUTO 300. See `BT_UNET_AUTO_300_REVISION.md` for source details and exploratory split warnings. External download success, GPU training and clinical performance remain unverified. True 3D CT/MRI and authentic spectral baseline integrations are still outstanding reviewer items; LinkNet is paper-aligned, not author-official. The 12-hour limit is per session, not for the full 324-entry campaign. Skipped jobs are not completed experiments.

## Validation

45 selected CPU tests passed, including actual resolved configurations for all 105 revised proposal jobs, LinkNet coverage, old-proposal rejection, unchanged-baseline reuse, mixed-plan paper merge, 300/165 compatibility, metric and model tests, and budget/recovery regression tests. No full training or remote downloads were performed.
