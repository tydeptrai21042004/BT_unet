# Kaggle experiment tools

For the updated **1-based 261-job, 11-hour Kaggle runner**, read
`docs/KAGGLE_11H_ONE_BASED_RUN_GUIDE.md` and use the supplied standalone
`BT_UNet_KAGGLE_11H.ipynb` notebook. The files `multi_t4_runner.py` and
`merge_sessions.py` are runnable CLI programs.

Plan all experiments without a GPU:

```bash
python scripts/kaggle/multi_t4_runner.py --scope ALL --one-based --start-index 1 --plan-only
```

Train/evaluate/ZIP starting from experiment #1 (Kaggle GPU + Internet required):

```bash
python scripts/kaggle/multi_t4_runner.py --scope ALL --one-based --start-index 1 --max-hours 11 --bundle-session
```

Resume at experiment #N using `BT_UNET_NEXT_EXPERIMENT_ALL.txt`, **not** the
legacy zero-based `next_start_index` field. See the guide for required data
sources, 3D/external baseline limitations and output retention.

## Kaggle non-interactive dataset fix (2026-11-17)

The runner no longer invokes `kagglehub.dataset_download()` for missing Kvasir-SEG. It prefers an already attached Kaggle Input, otherwise uses the registered official Simula HTTPS ZIP and, optionally, Kaggle CLI. For restricted notebook sessions, attach `debeshjha1/kvasirseg` via Add Input before selecting Save Version. A failed data setup keeps the 1-based experiment counter unchanged. See `docs/KAGGLE_11H_ONE_BASED_RUN_GUIDE.md`.

## AUTO dataset-only queue (no manual Kaggle dataset attachment)

The new `--scope AUTO` selects **153 experiments** needing only public direct
HTTPS dataset downloads. Use `BT_UNet_KAGGLE_11H_AUTO.ipynb` and
`docs/KAGGLE_AUTO_ONLY_GUIDE.md`. It skips optional expensive dataset audits
but still requires dataset preparation and a valid evaluation checkpoint.

```bash
python scripts/kaggle/multi_t4_runner.py --scope AUTO --one-based --start-index 1 --max-hours 11 --bundle-session
```
