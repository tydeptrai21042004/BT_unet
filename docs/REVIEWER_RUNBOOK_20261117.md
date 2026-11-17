# BSPC-D-26-14312 — executable reviewer revision guide

Commands run from the extracted repository root. **All experiments on real data must be run by the researcher**; this ZIP contains no medical datasets or pretrained external models.

## 0. Dependencies and tests

```bash
python -m pip install -r requirements.txt -r requirements/revision.txt
python -m pytest -q tests/test_review_patch_20261117.py tests/test_final_revision_package.py tests/test_revision_extensions.py
```

## 1. Patient/video-aware split audit

Provide `data/splits/kvasir_seg/{train,val,test}.txt`, or matching dataset-specific files; optionally `data/splits/kvasir_seg/clusters.csv` with columns `sample_id,cluster_id`. If no true cluster IDs exist, the audit **cannot** prove patient/video independence.

```bash
python scripts/revision/audit_splits.py --data-root data \
  --datasets kvasir_seg --compare-content-hashes \
  --output outputs_revision/audits/splits.json
```

Add `--cluster-map kvasir_seg=path/to/real_patient_video_map.csv` if needed. A mismatched train/val/test patient ID aborts the audit.

## 2. Matched-seed BT vs U-Net and factorial/residual ablations

```bash
# Plan only; inspect revision_plan.json before running.
python scripts/revision/run_full_revision.py \
  --groups primary,factorial,residual --datasets kvasir_seg \
  --seeds 42,1337,2024,7,9,11 --epochs 100 \
  --data-root data --output-root outputs_revision/main \
  --save-predictions --compare-content-hashes

# Same command plus --run --resume after confirming dataset and GPU.
```

For the 10-epoch concern, run separate `--groups convergence --datasets cvc_clinicdb` at 50 or 100 epochs, then use `plot_convergence_multiseed.py --runs ...` with the actual history.csv files. Never mix 10-epoch early-training performance with converged performance in one ranking without labels.

## 3. Seed-level statistical tests

Each candidate/reference CSV must aggregate the **same samples and the same set of independent training seeds** (concatenate compatible `per_image.csv` files; include seed column). For several independent contrasts, `--candidates` accepts several `MODEL=path.csv` entries.

```bash
python scripts/revision/statistical_comparison.py \
  --reference outputs_revision/aggregate/unet_per_image.csv \
  --candidates BT=outputs_revision/aggregate/bt_per_image.csv \
  --metric dice --bootstrap 3000 \
  --output outputs_revision/tables/statistics.csv
```

For clustered images, provide a `cluster_id` column in both CSVs and add `--cluster-field cluster_id`. Exact two-sided Wilcoxon with only three independent seeds cannot reach p<0.05.

## 4. Qualitative panels with GT-based zooms (>=300 dpi)

Run `evaluate_revision.py --save-predictions` for **each actual trained model**. Pass the `/masks` directory, **not** `/probabilities`.

```bash
python scripts/revision/plot_qualitative.py \
  --dataset kvasir_seg --data-root data --split test \
  --model-names 'U-Net,U-Net++,ResUNet++,PraNet,BT U-Net' \
  --pred-dirs 'outputs_revision/pred/unet/masks,outputs_revision/pred/unetpp/masks,outputs_revision/pred/resunetpp/masks,outputs_revision/pred/pranet/masks,outputs_revision/pred/bt/masks' \
  --ids 'REAL_TEST_ID_1,REAL_TEST_ID_2' --zoom --zoom-pixels 112 \
  --dpi 350 --output outputs_revision/figures/qualitative.pdf
```

Select cases **before comparing BT and baselines**, or use the deterministic reference-model selector. To label flat lesions, glare or clutter, supply verified annotations; an algorithm cannot infer clinical category solely from lesion size. Alongside every figure a JSON file logs mask and source-image SHA256.

## 5. Spectral interpretability and artifact sensitivity

```bash
python scripts/revision/analyze_spectrum.py \
  --checkpoint outputs_revision/main/kvasir_seg/primary/primary_dapr_unet/seed_42/primary_dapr_unet/checkpoints/best.pt \
  --output-dir outputs_revision/spectral

python scripts/revision/run_artifact_frequency_matrix.py \
  --checkpoint outputs_revision/main/kvasir_seg/primary/primary_dapr_unet/seed_42/primary_dapr_unet/checkpoints/best.pt --dataset kvasir_seg \
  --data-root data --corruptions glare,motion_blur,low_contrast \
  --severities 1,3 --bands none,low,middle,high \
  --output-root outputs_revision/artifacts
# Add --run after reviewing the planned evaluations.
```

These are input-domain corruptions and frozen-weight *interventions*, not a proof of clinical artifact suppression. The learned spectral response must be interpreted together with measured feature activations, not only the amplitudes of stored trainable coefficients.

## 6. Reproducible inference benchmarking

```bash
python scripts/revision/benchmark_inference.py \
  --checkpoint outputs_revision/main/kvasir_seg/primary/primary_dapr_unet/seed_42/primary_dapr_unet/checkpoints/best.pt --device cuda \
  --batch-size 1 --image-size 352 --warmup 30 --repetitions 200 \
  --output outputs_revision/efficiency/bt_cuda.json
```

Report median/p95 latency and the device. `conv_linear_macs` and `fft_butterfly_complexity_units` are separate quantities; **do not** label their sum as exact total FLOPs. FFT-aware exact FLOP counts need instrumented profiling or an explicit validated hardware-specific accounting methodology.

## 7. Official GFUNet/FDFUNet/AFD-Seg baseline requirement

The original project includes a **strict adapter**, not the full papers' architectures. Obtain authentic source from the authors and record its exact commit; do not present adapters/templates as official benchmark results.

```bash
python scripts/revision/install_official_source.py \
  --name gfunet --commit ACTUAL_UPSTREAM_GIT_SHA \
  --destination third_party/GFUNet
```

Complete `configs/controlled_spectral/AUDIT_TEMPLATE.json` in a **new** audit file: verified paper-to-code mappings, commit, pretrained dependencies, forward contract, preprocessing, train protocol. Then configure `configs/controlled_spectral/gfunet_OFFICIAL_TEMPLATE.yaml` with real module, class, source and output settings, and run:

```bash
python scripts/revision/validate_official_baseline.py \
  --config path/to/completed_gfunet.yaml \
  --output outputs_revision/official/gfunet_validation.json
```

Only after validation should `run_full_revision.py --groups official --external-configs path/to/completed_gfunet.yaml,...` be used. The original paper training protocol may need additional work beyond the unified BCE/Dice trainer. Official sources and weights are NOT bundled here.

## 8. New volumetric 3D CT/MRI extension

`src/models/proposal/bt_unet3d.py` defines `BTUNet3D` (full D/H/W FFT, direct spectral reconstruction) and `UNet3D` with the **same 3D encoder and decoder**. This is not the submitted 2D model.

Prepare `manifests_ct/train.csv`, `val.csv`, `test.csv` (also for MRI), each with:

```csv
sample_id,patient_id,image,mask
ct_001,patient_001,ct/images/ct_001.nii.gz,ct/masks/ct_001.nii.gz
```

NIfTI image/mask shapes, voxel spacing and affine coordinates must agree. Arrays `.npy` and `.npz` are supported for testing; `.npy` has no known physical spacing and cannot yield mm-based HD95. `.npz` must contain the `image` or `mask` key; a `spacing` array may provide (D,H,W) units if physically validated.

```bash
python scripts/revision/run_volumetric.py --model unet3d \
  --data-root data_3d --manifest-dir manifests_ct \
  --output-dir outputs_revision/ct/unet3d/seed42 \
  --patch 64 96 96 --epochs 100 --seed 42 --device cuda

python scripts/revision/run_volumetric.py --model bt_unet3d \
  --data-root data_3d --manifest-dir manifests_ct \
  --output-dir outputs_revision/ct/bt_unet3d/seed42 \
  --patch 64 96 96 --epochs 100 --seed 42 --device cuda
```

Repeat for at least three to six independent paired seeds on both the 3D U-Net and BT U-Net, and then repeat the same protocol on a real MRI dataset. To rerun evaluation without retraining, pass `--eval-only` with the matching `--model`, `--seed`, `--patch`, `--channels`, `--data-root`, `--manifest-dir`, and `--output-dir`. Training samples native 3D voxel patches; validation/test use overlapping 3D sliding windows to cover entire native volumes. All reported metrics are computed per full 3D patient volume. A NIfTI affine equality check is enforced. The implementation handles **binary segmentation only**; multiclass segmentation, modality-specific preprocessing, data augmentation and 3D medical backbones beyond U-Net are additional research scope, not silently claimed.

## 9. Manuscript-only reviewer comments

Revise the mathematical proposition descriptions, Introduction citations, abstract, title/keywords, and Figure 1 separately. This code-only patch cannot fix those written-document requests.
