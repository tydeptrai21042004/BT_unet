# Kaggle AUTO — 165 independent direct-download experiments

The standalone notebook `BT_UNet_KAGGLE_11H_AUTO.ipynb` starts a fresh GitHub clone and runs a fixed, independently numbered **165-experiment** plan. The original 261-job ALL queue is unchanged.

**Start with:** `docs/KAGGLE_AUTO_PREFLIGHT_AND_SKIP_GUIDE.md`, which documents every dataset, direct download URL, alternative Hyper-Kvasir mirror, limits of the live Kaggle preflight, skip handling, result files, and resume/merge commands.

AUTO keeps 153 existing jobs and adds 12 BUS-UCLM ultrasound experiments. Sources are tested by a lightweight 4-byte HTTPS ZIP probe **on Kaggle** before training. A missing dataset is recorded in `BT_UNET_SKIPPED_AUTO.json`, not treated as trained. The resume number comes from `BT_UNET_NEXT_EXPERIMENT_AUTO.txt` even if some earlier jobs were skipped.

```bash
python scripts/kaggle/multi_t4_runner.py --scope AUTO --one-based --plan-only
python scripts/kaggle/multi_t4_runner.py --scope AUTO --one-based --start-index 1 --max-hours 11 --bundle-session
```

Do not use the older 153-job counter with this 165-job plan, and never label a run with skipped jobs as all experiments completed. `--require-complete` detects skipped and missing experiments.
