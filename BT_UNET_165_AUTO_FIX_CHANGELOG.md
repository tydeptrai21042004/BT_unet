# 165 AUTO execution/export fix — verified 2026-11-18

Files changed relative to the uploaded `BT_unet-main(20261118-123911).zip`:

- `scripts/revision/evaluate_revision.py`: compute real, configuration-matched held-out test loss and report comprehensive per-image, boundary and pixel metrics; print results.
- `scripts/kaggle/multi_t4_runner.py`: use observed full-run timing and 1.08 planning factor (rather than 2.0), retain complete-epoch budget safety, create compact completed-results directories, bundle only lightweight outputs for AUTO, print each result, and disable prediction PNG dumps in AUTO. Non-AUTO archival remains intact.
- `scripts/kaggle/results_export.py` (new): verified compact per-experiment export, 165-row paper table CSV, completed CSV/JSON and copy-ready LaTeX cells.
- `scripts/kaggle/merge_sessions.py`: accept legacy nested-job ZIPs and new compact result directories.
- `scripts/kaggle/merge_paper_metrics.py` (new): merge legacy and new AUTO session ZIPs into a paper-ready 165-row CSV, with explicit mixed-commit approval.
- `BT_UNet_KAGGLE_11H_AUTO.py` and `.ipynb`: check new reporter marker; print all result artifact names.
- `tests/test_auto_paper_metrics_export.py` (new): evaluate test-loss correctness, metadata verification, compact ZIP integrity, merging and measured budget estimation.
- `KAGGLE_AUTO_165_PAPER_RESULTS_GUIDE.md` (new): Kaggle, restart, outputs and merge instructions.

Verification: original and patched `BT_UNET_PLAN_AUTO.csv` have identical SHA-256 `049981ce898bfae77fa28b026989428475c042682900d1def786b51d22d51125`. All new tests pass; full test suite reported 246 passed. Real CPU evaluation from an actual trained smoke checkpoint produced `test_loss` and all extra metrics.

Caveat: BUS-UCLM returned HTTP 403 during the user's previous Kaggle session. The code must skip unavailable training datasets rather than substitute them or invent results. A fully completed 165-run table still requires the real dataset to become accessible.
