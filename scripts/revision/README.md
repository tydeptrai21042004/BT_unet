# BT U-Net — executable BSPC revision experiments

This patch retains the submitted **direct-reconstruction BT U-Net** unchanged. Newly
registered models are explicitly labeled controls, not replacements for the
original architecture. No reported paper results or trained weights are bundled.

## Install and run tests

From the repository root, after overlaying the patch-only ZIP:

```bash
pip install -r requirements.txt -r requirements/revision.txt
python -m pytest -q tests/test_revision_extensions.py tests/test_dapr_unet.py tests/test_repository_scope_and_fairness.py
```

**Data prep:** prepare original datasets with `scripts/prepare_dataset.py` and
produce explicit `splits/<dataset>/train.txt`, `val.txt`, `test.txt` manifests.
Do **not** create manifests for external testing from a dataset already used
for model training. On public external test datasets, the `test.txt` manifest
can enumerate every independent test image, but must not be used in training.

## P0: matched residual and comprehensive BT component experiments

```bash
python scripts/benchmark_multi_seed.py \
  --models dapr_unet,dapr_residual_control,dapr_matched_residual,dapr_matched_residual_zero_init,dapr_matched_difference,dapr_amplitude_only,dapr_phase_only,dapr_mixing_only \
  --config-dir configs/revision --dataset kvasir_seg --data-root data \
  --seeds 42,1,2 --epochs 50 --batch-size 6 --image-size 352 \
  --output-root outputs_revision/controls --device auto
```

This reuses the original fair training framework, now aligning the cosine
`T_max` with a CLI epoch override. Default checkpoint retention is on.
Review existing model-specific config fields before running each comparison.

For the manuscript's original five-component ablation:

```bash
python scripts/run_dapr_ablation.py --dataset kvasir_seg \
  --data-root data --seeds 42,1,2 --no-delete-checkpoints-after-eval
```

## P0: cross-dataset generalization

Prepare both source datasets and all external targets, and make explicit
split manifests. Source train/val/test disjointness is checked; use
`--compare-content-hashes` to detect exact duplicate images.

```bash
python scripts/revision/run_cross_dataset.py \
  --sources kvasir_seg,cvc_clinicdb --targets cvc_colondb,etis,cvc_300 \
  --val-source kvasir_seg --models unet,unetpp,resunetpp,dapr_unet \
  --seeds 42,1,2 --epochs 50 --data-root data --dry-run
# Drop --dry-run to run after checking the commands and manifests.
```

This trains from SOURCE `train` files only and selects checkpoints using a
SOURCE `val` file only. Every external target is evaluated separately on its
`test` manifest. These results are **not** the same protocol as the original
within-dataset experiments. Preserve patient/video grouping where available.

## P0: per-image masks, boundaries, and reviewer-ready qualitative figures

```bash
python scripts/revision/evaluate_revision.py \
  --checkpoint outputs_revision/controls/seed_42/dapr_unet/checkpoints/best.pt \
  --dataset kvasir_seg --data-root data --save-predictions \
  --output-dir outputs_revision/eval/bt

python scripts/revision/plot_qualitative.py \
  --dataset kvasir_seg --data-root data \
  --model-names 'U-Net,U-Net++,ResUNet++,PraNet,BT U-Net' \
  --pred-dirs 'outputs_revision/eval/unet/masks,outputs_revision/eval/unetpp/masks,outputs_revision/eval/resunetpp/masks,outputs_revision/eval/pranet/masks,outputs_revision/eval/bt/masks' \
  --ids 'YOUR_TEST_SAMPLE_ID_1,YOUR_TEST_SAMPLE_ID_2' \
  --output outputs_revision/figures/qualitative.png
```

Prediction folders for baseline models must be generated from their own
checkpoints first. Red overlay: false positive; cyan: false negative; lime:
predicted boundary. Image-based metrics are measured at the transformed
resolution; HD95 and ASSD are in **pixels**, not physical units. Empty mask
conventions are documented in `src/analysis/boundary_metrics.py`.

## P0: frequency interpretation and interventions

```bash
python scripts/revision/analyze_spectrum.py \
  --checkpoint outputs_revision/controls/seed_42/dapr_unet/checkpoints/best.pt \
  --output-dir outputs_revision/spectral

python scripts/revision/run_frequency_interventions.py \
  --checkpoint outputs_revision/controls/seed_42/dapr_unet/checkpoints/best.pt \
  --dataset kvasir_seg --data-root data \
  --output-root outputs_revision/band_interventions
```

The heatmaps are checkpoint-based learned responses. The operator Frobenius
map is not a measured clinical denoising result. Frequency interventions remove
low/mid/high rFFT coefficient bands **at inference** and report actual Dice
drops. They are mechanistic sensitivity checks, not causal medical explanations.

## P0: statistics and convergence

```bash
python scripts/revision/statistical_comparison.py \
  --reference outputs_revision/eval/unet/per_image.csv \
  --candidates BT=outputs_revision/eval/bt/per_image.csv \
  --bootstrap 2000 --output outputs_revision/tables/stats.csv

python scripts/revision/plot_convergence.py \
  --runs U-Net=outputs_revision/controls/seed_42/unet/results/history.csv \
         BT=outputs_revision/controls/seed_42/dapr_unet/results/history.csv \
  --output outputs_revision/figures/convergence.png
```

The statistics script performs strict sample-ID/seed matching and reports
bootstrap intervals and **exploratory** paired-t p-values. For group-dependent
images, add a real patient/video grouping column and `--cluster-field`. Do not
claim patient-level significance using image-level pairs.

## P1: corruption robustness and limited annotation

```bash
python scripts/revision/run_robustness.py \
  --checkpoint PATH/TO/best.pt --dataset kvasir_seg --data-root data \
  --output-root outputs_revision/robustness

python scripts/revision/make_low_data_splits.py \
  --dataset kvasir_seg --data-root data --fractions 0.1,0.25,0.5,1.0 \
  --seed 42 --output-dir outputs_revision/splits

python scripts/train_one.py --model dapr_unet --dataset kvasir_seg \
  --config configs/fair/dapr_unet.yaml --data-root data \
  --train-split-file outputs_revision/splits/train_025pct_seed42.txt \
  --epochs 50 --output-root outputs_revision/low_data_25
```

All corruption masks are unchanged and deterministic for each sample ID,
severity and seed. The fraction files are nested subsets of the **training**
manifest; validation and test files remain unchanged. Train each comparison
model using exactly the same subset manifest.

## P1: synchronized efficiency

```bash
python scripts/revision/benchmark_inference.py --checkpoint PATH/TO/best.pt \
  --image-size 352 --batch-size 1 --device auto \
  --output outputs_revision/efficiency/bt.json
```

GPU results synchronize before/after each timed batch. Record your device and
precision. CPU timings should not be compared with the submitted Tesla T4 FPS.

## Contemporary published spectral baselines

`src/models/baselines/external_spectral.py` is an **adapter**, not a fake
GFUNet/FDFUNet/AFDSeg implementation. Obtain the selected method's genuine
source and license separately, audit its pretraining/output/loss assumptions,
and make its Python class importable via `PYTHONPATH`. Register it through:

```yaml
model:
  name: external_spectral
  module_path: official_package.module
  class_name: OfficialNetworkClass
  model_kwargs:
    in_channels: 3
    num_classes: 1
```

The model's actual argument names must come from its verified implementation;
its logits and resolution must satisfy the benchmark interface. Record the
original code commit, pretrained weights policy and architectural adaptations.
No unverified method or reported performance is included in this patch.

## Reproducibility/limitations

- No training results, significance claims, spectral plots, published-model
  reproductions, or numerical reviewer responses are invented.
- The dataset loader in the original repository silently loads all images when
  a split is absent; the new revision tools **fail closed** on missing manifests.
- Output checkpoints are kept by default in the modified Kaggle scripts. To
  recover the old deletion behavior in those scripts, set `KEEP_CHECKPOINTS=0`.
- Direct BT's Fourier computations, output projection and decoder are unchanged.
- Existing benchmark results should not be overwritten: use separate
  `outputs_revision/...` directories.
