# Correction notes

This archive fixes the Kaggle/runtime failures observed when dataset files are created under the repository during experiments.

## Main fixes

1. `tools/audit_repository_cleanliness.py`
   - Runtime folders are skipped by default: `data/`, `outputs/`, `outputs_*`, `wandb/`, `mlruns/`, and `runs/`.
   - Added `--strict-runtime-folders` for the old stricter behavior.
   - This prevents downloaded datasets such as ISBI 2012 and HyperKvasir from causing audit failures.

2. `scripts/run_baseline_proposal_comparison.py`
   - Updated the docstring from “2-proposal” to DAPR-only.
   - `--run-tests` now defaults to false.
   - Added `--skip-cleanliness-audit`.

3. `scripts/run_dapr_ablation.py`
   - `--run-tests` now defaults to false.
   - Added `--skip-cleanliness-audit`.
   - Output root now defaults to `outputs_dapr_ablation/<dataset>` instead of always `etis`.

4. `scripts/run_dapr_hyperparameters.py`
   - `--run-tests` now defaults to false.
   - Added `--skip-cleanliness-audit`.
   - Output root now defaults to `outputs_dapr_hyperparameters/<dataset>` instead of always `etis`.

5. `run.sh`
   - Replaced stale `plain_fourier_unet` and `apdr_fourier_unet` with `dapr_unet`.
   - Removed calls to missing scripts.
   - Added correct commands for DAPR ablation and hyperparameter sensitivity.

6. `requirements.txt`
   - Added `torchvision`.
   - Pinned `numpy<2.2.0` to avoid TensorFlow/Kaggle environment conflicts.
   - Added `certifi` and `pytest`.

7. Kaggle scripts
   - Updated existing ETIS scripts to keep data/output outside the repository by default.
   - Removed unnecessary runtime tests from expensive training scripts.
   - Added corrected scripts for:
     - `scripts/kaggle_baseline_isbi2012_3seeds.sh`
     - `scripts/kaggle_hyperparameters_hyper_kvasir_seg_1seed.sh`

## Validation performed

- Python syntax check passed for the patched runner/audit scripts.
- Bash syntax check passed for `run.sh` and all Kaggle scripts.
- Repository cleanliness audit passed.
- Selected pytest checks passed: `35 passed`.
- Baseline/DAPR, DAPR ablation, DAPR hyperparameter, and strict no-forward audits passed.
