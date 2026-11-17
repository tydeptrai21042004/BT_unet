# BSPC-D-26-14312 — reviewer-to-code traceability (2026-11-17)

This is a **code capability** audit, not a claim that experiments were run. The submitted 2D BT U-Net equations, registered model (`dapr_unet`), training baselines, and checkpoint schema were **not changed** by this patch. The supplied archives `075735` and `083100` were binary-identical across all 234 files.

## Requirements and deliverables

| Reviewer comment | Implemented path | Status of code | Evidence still required |
|---|---|---|---|
| R1 CT/MRI volumetric | `src/models/proposal/bt_unet3d.py`, `src/datasets/medical_volume_dataset.py`, `scripts/revision/run_volumetric.py` | **New** 3D BT/3D U-Net; patch training; patient split audit; full-volume sliding window; Dice/IoU/HD95 | Genuine CT and MRI datasets, splits, multiple-seed results, training hardware and checks |
| R1 spectral model SOTA; R2.3; R5.1 | `src/models/official/strict_external.py` (existing), `install_official_source.py` (existing), `validate_official_baseline.py` (new) | **Fail-closed interface only**; NO upstream GFUNet/FDFUNet/AFDSeg source/weights supplied | Pin actual official source commit, audit, dependencies, run actual comparison; do not call templates models |
| R1 transformer SOTA | Existing `src/models/baselines/polyp_pvt.py` | Local implementation remains; no verified new official transformer | Additional sourced state-of-art transformer model and fair benchmark if requested |
| R1 patient/video leakage / R2 protocol | `audit_splits.py` + `run_full_revision.py` | Explicit dataset-specific manifests, optional patient/video cluster mappings, SHA256 overlap checks, run provenance | Actual patient IDs/volumes; run split audit on real data |
| R1 statistics; R2.6 | `statistical_comparison.py` | Seed-mean paired tests, Wilcoxon+paired-t and Holm adjustments; cluster bootstrap | >=6 distinct paired seeds, per-image CSVs from true checkpoints; report actual p values |
| R1 efficiency | `benchmark_inference.py` | Measured params, synchronized median/p95 latency, FPS, CUDA allocated/reserved memory; counted conv/linear MACs; FFT complexity separate | Actual trained checkpoints, hardware; exact total FLOPs require instrumented FFT-aware profiling |
| R1 difficult/failure cases, R2.1, R3, R7 300-dpi | `plot_qualitative.py`, `select_qualitative_cases.py` | Deterministic GT-based zoom crop, masks, FP/FN overlays, 350 dpi, PDF supported | Prediction masks from U-Net, U-Net++, ResUNet++, PraNet, BT; separately verified case annotations |
| R2.2 spectral responses | `analyze_spectrum.py` | 350-dpi PNG+PDF of amplitude, phase and operator gain; low/mid/high band CSV | Trained model, frequency/activation responses; avoid treating gain maps as causal artifact mechanisms |
| R2.2 glare/blur artifact effect | `run_artifact_frequency_matrix.py` | Crosses fixed checkpoint, corruption, severity, intervention, produces Dice drops | Actual evaluation results on real images |
| R2.4 training details | `train_one.py`, config files, training manifests | Original implementation has AdamW, cosine, BCE+Dice; revised runner records config and manifest hashes | Report table-specific effective settings in paper |
| R2.5 convergence | `run_full_revision.py --groups convergence`, `plot_convergence_multiseed.py` | Multi-seed training/curve tools, 350-dpi export | Train 50–100 epochs and report curves/metrics |
| R2.7 residual baseline | `configs/revision/*`, `run_full_revision.py --groups residual` | Matched residual and original control variants with alpha sweep; corrected each variant config path | Real multi-seed ablation table |
| R5.2 channel mixing contribution | `run_full_revision.py --groups factorial` | Factorial controls and multi-seed plan (fixed missing-config bug) | Actual n>=3 or 6 results for channel-mix contribution |
| R2 propositions, R5 minor and R7 language | Manuscript only | **No code modification required** | Revise written claims, figures, references and format separately |

**Status rule:** A runnable code path is *not* experimental evidence. The ZIP deliberately contains no fake checkpoint, no claimed official model implementation, and no fabricated CT/MRI, statistical or clinical results.

## Changed code relative to 083100

New:
- `src/models/proposal/bt_unet3d.py`
- `src/datasets/medical_volume_dataset.py`
- `scripts/revision/run_volumetric.py`
- `scripts/revision/run_artifact_frequency_matrix.py`
- `scripts/revision/validate_official_baseline.py`
- `tests/test_review_patch_20261117.py`
- `docs/REVIEWER_CODE_MAPPING.md`
- `docs/REVIEWER_RUNBOOK_20261117.md`

Updated:
- `scripts/revision/audit_splits.py`
- `scripts/revision/run_full_revision.py`
- `scripts/revision/plot_qualitative.py`
- `scripts/revision/analyze_spectrum.py`
- `scripts/revision/benchmark_inference.py`
- `scripts/revision/statistical_comparison.py`
- `scripts/revision/plot_convergence.py`
- `scripts/revision/plot_convergence_multiseed.py`
- `requirements/revision.txt`
- `tests/test_repository_scope_and_fairness.py` (reconciled previously stale exact-structure assertions)

Not changed:
- `src/models/proposal/fourier_unet.py`
- `src/models/proposal/dapr_unet.py`
- `scripts/train_one.py`
- All original `configs/fair/` and published experiment tables.
