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
