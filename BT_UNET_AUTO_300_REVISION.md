# BT U-Net — append-only 300-experiment revision (historical)

The active notebook now uses AUTO 324. See [the current guide](BT_UNET_AUTO_324_REVISION.md) for revised proposal reruns and archive reuse. The instructions below describe the legacy 300 plan.

Base repository: `0cbf0b01d56722489276ee2d6c341df23aabb3c6`.

## Schedule

The existing AUTO-165 manifest is unchanged (all original job dictionaries, IDs, seeds, epochs, and order). `--extend-300` appends 135 training entries. No reported scores are fabricated or included as completed results in this repository.

| Entries | Dataset/group | New jobs | Epochs |
|---|---|---:|---:|
| 1–165 | Original manifest | 0 | unchanged |
| 166–192 | CVC-ClinicDB | 27 | 100 |
| 193–219 | CVC-ColonDB (380-image benchmark) | 27 | 50 |
| 220–246 | ETIS-LaribPolypDB | 27 | 50 |
| 247–273 | ISIC 2018 Task 1 labeled training images | 27 | 30 |
| 274–288 | LinkNet on the five original datasets | 15 | match legacy reference duration |
| 289–300 | Hermitian-constrained BT on four datasets | 12 | match corresponding original BT duration |

All additions have seeds 42, 1337, 2024, 352-pixel inputs and batch size 6. Each new dataset has U-Net, U-Net++, Attention U-Net, ResUNet++, PraNet, HarDNet-MSEG, Polyp-PVT, LinkNet and original BT U-Net. LinkNet is a ResNet-18, no-pretraining, paper-aligned binary reimplementation (not author-official code), following Chaurasia and Culurciello, VCIP 2017, DOI 10.1109/VCIP.2017.8305148, https://github.com/e-lab/pytorch-linknet . It is a genuine additional published architecture, not a spectral baseline. Official GFUNet/AFD-Seg comparison and true 3D CT/MRI experiments remain separate reviewer requirements.

## Run on Kaggle

1. Push the updated repository to GitHub, then use `BT_UNet_KAGGLE_11H_AUTO.ipynb` (one code cell). Enable internet and GPU; pin the new commit SHA. Default `FIRST_EXPERIMENT=166`, `LAST_EXPERIMENT=300`, `SESSION_HOURS=12` avoids accidentally rerunning the original queue. Set FIRST to the printed next experiment number on a new session.
2. Attach prior `BT_UNET_SESSION_AUTO_*.zip` packages as Kaggle Input. Default `REUSE_ARCHIVE_DIR=/kaggle/input` scans prior packages, validates exact prefix manifest, row identity, full experiment spec and checkpoint/test provenance, skips matching completed runs, and recovers failure rates from retained per-image CSV when possible. These are identity/provenance checks, not independent proof of identical splits across older sessions. Missing original weights cannot be recovered from metrics alone.
3. To recover missing entries among 1–165, start a **fresh** session with `FIRST_EXPERIMENT=1`, `LAST_EXPERIMENT=165` and mount old compact archives. If the archives are unavailable, use targeted ranges instead (e.g. 11–12, 19–19, 58–105) after checking your own evidence. Existing skipped rows are not completed. Do not run all old entries without archive reuse unless intentionally regenerating them.
4. For plan inspection only: `python scripts/kaggle/multi_t4_runner.py --scope AUTO --extend-300 --plan-only`. Direct `--scope AUTO` without the extension flag remains the original 165-run interface.
5. For local CSV consolidation, collect both old and new session ZIPs in one directory and run `python scripts/kaggle/merge_paper_metrics.py --input-dir results --output-dir merged --allow-mixed-commits`. This explicit opt-in allows the append-only plans after prefix verification; retain source packages. The older `merge_sessions.py` remains strict and is not intended for overlapping/reused mixed-plan intervals. Conflicting duplicate scores are rejected.

## Dataset acquisition

Original five datasets remain. The new sources use public archives/Kaggle mirrors with fallback and exact pair-count validation; there is no synthetic fallback. Source selection and split provenance are recorded. Live full downloads have not been tested in this offline validation environment. Public providers may throttle, change layout, require authorization, or fail; failures are logged as skipped and remain retryable. The full official ISIC archive is large; the compact MedSegDB mirror is tried first. Downloaded archives are not bundled in this code ZIP.

| Dataset | Expected pairs | Sources |
|---|---:|---|
| Kvasir-SEG | 1000 | existing Simula HTTPS |
| Hyper-Kvasir SEG | existing protocol | existing Simula/HF HTTPS |
| Kvasir-Instrument | 590 | KaggleHub, Simula, MedSegDB, Kaggle HTTPS |
| ISBI2012 | 30 labeled slices | existing ImageJ HTTPS |
| BUS-UCLM | 683 | HF dataset, Kaggle HTTPS, KaggleHub; validates 38 patient IDs, 419 empty masks |
| CVC-ClinicDB | 612 | Kaggle balraj98/cvcclinicdb, MedSegDB CVC_ClinicDB.zip, Kaggle orvile/cvc-clinicdb |
| CVC-ColonDB | 380 | Kaggle longvil/cvc-colondb, existing PraNet test-bundle Google Drive fallback |
| ETIS | 196 | Kaggle nguyenvoquocduong/etis-laribpolypdb, existing PraNet test-bundle fallback |
| ISIC2018 | 2594 | MedSegDB ISIC2018.zip, official ISIC 2018 input + masks archives |

Public source documentation: https://huggingface.co/datasets/medicalai/MedSegDB/tree/main ; https://challenge.isic-archive.com/data/ ; https://www.nature.com/articles/s41746-026-02345-7 ; https://data.mendeley.com/datasets/7fvgj4jsp7/3 . Respect original dataset licenses and cite original authors. The 380-image ColonDB benchmark is not silently equated with the 300-frame original release. ClinicDB, ColonDB and ETIS use deterministic contiguous **exploratory** image splits by default, NOT verified patient/video-separated splits. Provide `BT_SPLITS_DIR` containing reviewed group splits for publication-grade analysis. ISIC uses existing generic seed-42 splits of its labeled training archive; lesion/patient separation is not independently verified. Do not claim cross-dataset independence without duplicate/overlap auditing.

## Minimal mathematical change

The original default is unchanged. `configs/fair/dapr_hermitian.yaml` enables `fourier_hermitian_constraint: true` without adding learned parameters. On the stored rFFT zero and Nyquist columns (only when spatial width is even), replace the positive gain by the geometric mean of paired gains and the phase by its antisymmetric average. The real mixer is unchanged. This enforces conjugacy of the constrained Fourier response; the 12 new runs test rather than assume performance improvement. Original model weights and scores remain labeled separately. Do not pool original and constrained BT results as if the same method. `analyze_spectrum.py` additionally reports Hermitian error and mixer singular-value/condition estimates, with the lower-bound statement restricted to the constrained spectral core.

## Reporting and evaluation

New runs automatically report 24 segmentation/probability fields: legacy Dice, IoU, precision, recall, MAE, loss, boundary F1, HD95, ASSD, pixel accuracy, specificity, balanced accuracy, F2; plus surface Dice at the boundary tolerance, MCC, foreground area error, Brier, NLL, ECE, predicted-foreground ECE, average precision (noninterpolated AUPRC), complete-miss rate among positive samples, empty-prediction rate and undefined-distance rate. Undefined quantities are not substituted by invented values. Per-image confusion counts and defined counts are preserved. The evaluator computes all these from its existing prediction loop. Evaluation summary includes actual forward throughput, trainable parameters and GPU peak allocation; a separate standardized batch-1 benchmark is run for seed-42 ClinicDB configurations (optional failure does not invalidate training). Conv/linear MACs and FFT complexity are reported separately. Results include `BT_UNET_EFFICIENCY_AUTO.csv` for current compact records.

The existing checkpoint-only evaluator, spectral, robustness, frequency-intervention, statistical and qualitative scripts can reuse selected best checkpoints. Selected main checkpoints (seed-42 standard models, all U-Net/BT/LinkNet/ResUNet++ standard runs, and convergence runs) and split manifests are retained; no duplicate training is needed to calculate metrics if suitable old weights or predictions exist. No claim is made that every optional post-hoc experiment has been executed or scheduled among the 300 training jobs. To evaluate an old checkpoint, prepare the exact original split and run:

```
python scripts/revision/evaluate_revision.py --checkpoint path/to/best.pt --dataset kvasir_seg --data-root data --split test --image-size 352 --output-dir results/recovered
python scripts/revision/benchmark_inference.py --checkpoint path/to/best.pt --output results/efficiency.json
python scripts/revision/analyze_spectrum.py --checkpoint path/to/bt_best.pt --output-dir results/spectrum
```

## Validation

61 selected CPU regression tests passed (queue identity, Hermitian even/odd grids and gradient propagation, old checkpoint loading, LinkNet forward/backward, metric formulas, archive import, legacy export/recovery, dataset helpers). No GPU training or full remote dataset downloads were conducted. The 300 entries are planned, not completed; previous 114 manuscript entries can only be skipped automatically if their matching archive evidence is provided. No new accuracy values are claimed.
