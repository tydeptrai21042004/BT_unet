# BT U-Net — 12-hour AUTO recovery patch

Extract the patch at the repository root, replacing the matching files. Commit and push. The existing `BT_UNet_KAGGLE_11H_AUTO.py` and `.ipynb` filenames are deliberately retained for compatibility; both now default to **12.0 hours**. Enable Internet and a GPU in Kaggle. No manual dataset attachment is required for the normal acquisition path.

## Recovery ranges

Edit `FIRST_EXPERIMENT` and `LAST_EXPERIMENT` in the AUTO notebook (inclusive):

| First | Last | Purpose |
|---:|---:|---|
| 10 | 12 | 100-epoch U-Net++ |
| 58 | 93 | Kvasir-Instrument |
| 94 | 105 | BUS-UCLM |

Default 1–165 remains the complete original plan. For other missing runs, choose their contiguous intervals. Do not overwrite verified previously completed results. The existing merger expects contiguous session coverage; keep prior session archives, and use per-run exports to reconcile skipped intervals rather than assuming a skipped row is complete.

## Automatic data acquisition

- Kvasir-Instrument: KaggleHub `debeshjha1/kvasirinstrument`, original Simula HTTPS archive, then MedSegDB HTTPS mirror. Existing appropriately named Kaggle inputs are also discovered. Validate 590 paired decodable images/masks and dimensions.
- BUS-UCLM: Hugging Face `MedOtter/BUS-UCLM` (683 records, binary masks and patient IDs), then KaggleHub `orvile/bus-uclm-breast-ultrasound-dataset`. The old Mendeley V1 S3 link is removed. Validate 683 pairs, 38 patient prefixes, 419 empty masks; the HF import additionally checks 174 benign / 90 malignant / 419 normal labels. Preserve binary foreground. Reject mismatched image/mask dimensions rather than guessing alignment.
- Sources are tried lazily when needed. Validation failure tries another candidate; if all sources fail for either dataset, the current experiment remains pending and the scheduler stops with diagnostics. Source receipt is written to the job provenance. No substitute datasets or fabricated scores.
- BUS-UCLM uses patient-level splits (29 training, 4 validation, 5 test patients). Test IDs are the authors' suggested COPE, ANFO, ELCO, CRCI, FLKA; validation patients are selected deterministically with seed 42. This is a deliberate split-protocol improvement for the previously unexecuted dataset; update the manuscript accordingly. Other split rules stay unchanged. Source pages: https://datasets.simula.no/kvasir-instrument/ ; https://huggingface.co/datasets/MedOtter/BUS-UCLM ; https://constantinpape.github.io/torch-em/torch_em/data/datasets/medical/bus_uclm.html .

## Time budget and full-state recovery

The 12-hour cap includes startup, setup, training, evaluation and packaging. A cleanup reserve (20 minutes at full budget) and evaluation reserve remain inside the 12 hours. AUTO calibrates one epoch (including recovery checkpoint I/O), or reuses a compatible measured estimate. Full planned epochs multiplied by the per-epoch estimate and 1.08, plus evaluation and cleanup reserves, must fit in the remaining session time. Otherwise the scheduler defers the job before full training and preserves the experiment number. It does not intentionally start a partial run. The gate conservatively checks the full configured duration even if a recovery checkpoint exists; this avoids admitting an over-budget run. Resumed wall times are not used as full-run timing estimates. Each completed epoch writes an atomic full-state checkpoint in `/kaggle/working/BT_UNET_RECOVERY/<experiment_id>.pt`. The current epoch may be lost if interrupted, but the previous complete epoch survives. Optimizer, scheduler, scaler, best state, history, RNG and configuration/split/content signature are restored. Legacy `--resume` remains weights-only; new `--recovery-path` is full-state resume.

If Kaggle ends the session, keep the recovery `.pt` from notebook output; in a fresh session attach the previous notebook output or a dataset containing that exact file. The runner searches `/kaggle/input` for it and copies it automatically. Set `FIRST_EXPERIMENT` to the unchanged pending number. Same working directory recovery is automatic. Use trusted checkpoints only. A new session cannot recover files that Kaggle has not persisted. Completed recovery files are deleted after successful export; compact result ZIPs still omit checkpoints. An individual experiment that is estimated to exceed even a fresh 12-hour session will remain deferred until a sufficiently fast environment is available; an admitted experiment can still run longer than predicted. Checkpoints may consume hundreds of MB; save them separately from the compact metrics ZIP when needed.

## Validation

Offline regression tests cover source fallback, pair validation, original queue IDs, and full-state recovery. Run:

```bash
python -m pytest -q tests/test_auto_12h_recovery.py tests/test_kaggle_auto_dataset_preflight.py tests/test_bus_uclm_dataset_support.py tests/test_kaggle_auto_queue.py tests/test_scheduler_tmax_alignment.py
```

No live Kaggle GPU timing or end-to-end dataset downloads were executed during patch creation. Sources, schemas, and dataset counts were checked against public documentation, not cryptographically matched to the original data. A remote outage or unexpected archive layout may still prevent acquisition. Model code, training precision, epochs, resolutions and experiment IDs remain unchanged.
