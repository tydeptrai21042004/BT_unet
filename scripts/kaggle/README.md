# Kaggle T4 multi-job runner

This repository addition contains only the sequential scheduler. **No models or existing training scripts are edited.** The standalone notebook bootstrap clones GitHub, checks the repository commit, and executes `scripts/kaggle/multi_t4_runner.py`.

1. Copy the contents of the GitHub patch ZIP to the root of the BT_unet repository, preserving paths, then commit and push.
2. In Kaggle, select GPU T4, turn on Internet, and paste `BT_UNet_RUN_FROM_GITHUB_T4.py` into one Python cell. For the first run leave `SCOPE='CORE'`, `START_INDEX=0`.
3. Download all produced `BT_UNET_RESULTS__*.zip` and the `BT_UNET_NEXT_CORE.json` status after each session. If stopped by projected runtime, the previous completed ZIPs remain valid.
4. For the next session set `START_INDEX` to `next_start_index` from JSON and set `PINNED_COMMIT` to the first session's `[SOURCE COMMIT]`. Do not restart the queue from zero without intending to repeat runs.
5. After CORE ends at index 129, switch to `SCOPE='OPTIONAL'` and set `START_INDEX=0` for the 132 optional jobs.

The queue preserves all source experiment definitions: 129 CORE + 132 OPTIONAL, fixed 30/50/100 epochs, 352px images, batch size 6. The runner calibrates 1 epoch per *distinct* config, applies a conservative 2x timing factor, reserves time for evaluation and packaging, and never starts training a job it estimates cannot finish. It prints an activity heartbeat during long subprocesses. It cannot guarantee wall time if dataset discovery inside dependencies hangs or Kaggle cuts sessions early. Training that exceeds the predicted budget is marked failed, not completed.

The script will prepare datasets, including Kvasir-SEG through Kaggle's API (with certificate checks enabled). For other datasets with inaccessible source sites, attach originals as Kaggle Input and set `BT_DATA_<DATASET>_SOURCE` or `BT_DATA_<DATASET>_ZIP` in your cell. Dataset splits are generated with seed 42 by default; supply reviewed manifests with `BT_SPLITS_DIR` if needed. Separate sessions **must reuse the same Git SHA, dataset content, and split protocol** for a controlled comparison.

Results are written as `BT_UNET_RESULTS__<job id>.zip` and archived only after successful training + evaluation. A partially written ZIP has suffix `.partial` and cannot be mistaken for completed results. The status file `BT_UNET_NEXT_<SCOPE>.json` gives the next index. Use your original POSTPROCESS aggregator with these result ZIPs. There is no genuine mid-epoch resume in the existing `train_one.py` implementation. Baselines are not certified official-source reproductions by this scheduler.
