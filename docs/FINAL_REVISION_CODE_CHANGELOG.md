# BT U-Net reviewer revision — code change record

## Added and tested in this ZIP

- **Factorial controls:** `dapr_identity_spectral_control`, and the seven existing
  amplitude/phase/mixing combinations. The original `dapr_unet` default algorithm
  and forward result are unchanged.
- **Spatial-mixer control:** `dapr_spatial_channel_mix_control` relocates the same
  learned frequency-independent 1x1 channel mixing after the inverse FFT. This
  is mathematically equivalent to pre-IFFT mixing for direct reconstruction;
  tests verify numerical agreement with identical weights.
- **Residual sweep:** experiment matrix supports alpha=0.25, 0.50, 1.00 with
  normal/zero initialization and matched difference residual. Existing residual
  model classes are reused rather than inventing replacements.
- **Multi-seed orchestrator:** `scripts/revision/run_full_revision.py` plans and
  optionally executes factorial, residual, and convergence runs with separate
  configurations, run identifiers, seeds, best checkpoints, and per-image metrics.
  Dry-run is the default; `--run` explicitly starts training.
- **Full-convergence analysis:** new `plot_convergence_multiseed.py` computes
  mean and standard-deviation envelopes from actual epoch histories.
- **Mechanism evidence:** new `analyze_feature_spectrum.py` captures projected
  and reconstructed bottleneck activations and calculates frequency energy bands.
- **Segmentation analysis:** new `analyze_object_size.py` and
  `select_qualitative_cases.py` for predefined size categories and reproducible
  qualitative case selection.
- **Manuscript tables:** `build_manuscript_tables.py` aggregates per-image Dice,
  IoU, boundary metrics and failure counts per training seed, then across seeds.
  Missing runs cause a failure by default instead of being filled in.
- **Evaluation fixes:** output `seed` now matches the actual training checkpoint
  seed; `corruption_seed` is separate. Explicit empty-prediction, complete-miss
  and undefined-distance reporting was added.
- **Statistics fix:** paired significance tests use independent per-seed mean
  differences, not individual images masquerading as training replicates.
  Optional patient/video cluster bootstrap remains distinct from training-seed
  uncertainty. Three-seed exact Wilcoxon p-values cannot be below .05.
- **Baseline fidelity:** source-pinning script and `verified_external_spectral`
  fail-closed adapter refuse to use missing/modified/unaudited upstream code,
  and require explicit original output contracts. Architecture mappings must
  reference actual pinned source files.

## NOT claimed as implemented/verified

GFUNet and AFDSeg **original full model code is not bundled**. The development
runtime had no GitHub network access. Do not call the templates runnable baselines
until exact source clones, environment/dependencies, paper-to-code audit, and
paper-faithful official training settings have been independently verified.
No new training results, trained weights, GFUNet/AFDSeg benchmark numbers,
statistical significance conclusions, or clinical performance claims are included.

## Validation performed

- Targeted automated tests passed: see `tests/test_final_revision_package.py`
  and `tests/test_revision_extensions.py`.
- A one-epoch tiny *synthetic* dataset smoke check validated train_one.py,
  checkpoint saving and evaluate_revision.py for the identity ablation. These
  synthetic results are not medical benchmark results.
- Full pre-existing repository test suite includes computationally expensive
  model checks and was not fully completed in this runtime.
