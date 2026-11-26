# BT U-Net benchmark repository

This repository keeps one active proposal method:

> **DAPR U-Net: Direct Amplitude--Phase Reconstruction U-Net**

DAPR replaces the deepest U-Net feature with a directly reconstructed complex Fourier representation. It does **not** use the older residual Fourier proposal, APDR proposal, or DAPR-BAF boundary-refinement proposal as active manuscript methods.

## Active fair comparison

The fair benchmark compares eleven controlled baseline implementations against DAPR U-Net:

```bash
python scripts/run_baseline_proposal_comparison.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seeds 42,1,2 \
  --config-dir configs/fair
```

The output table is written to:

```text
outputs_baseline_proposal/etis/results/tables/baseline_proposal_comparison.tex
```

## DAPR component ablation

Run this to verify which parts of DAPR matter:

```bash
python scripts/run_dapr_ablation.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seeds 42,1,2
```

This suite contains:

- `dapr_residual_control`
- `dapr_no_global_phase`
- `dapr_no_global_amplitude`
- `dapr_no_global_channel_mix`
- `dapr_unet`

## DAPR hyperparameter sensitivity

A hyperparameter-sensitivity table is recommended because DAPR has Fourier-specific knobs. Run:

```bash
python scripts/run_dapr_hyperparameters.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seeds 42,1,2
```

This checks expansion ratio, amplitude scale, phase bound, and Fourier dropout under the same training and evaluation protocol.


## BUS-UCLM DOI-backed dataset

BUS-UCLM is registered as `bus_uclm` and can be prepared from the official Mendeley Data archive without Kaggle:

```bash
python scripts/prepare_dataset.py \
  --dataset bus_uclm \
  --data-root data \
  --image-size 352 \
  --force
```

See `docs/BUS_UCLM_GUIDE.md` for the DOI, official source, RGB mask conversion rule, split command, and efficiency-table benchmark command.

## Audit tools

Before launching expensive experiments, run:

```bash
python tools/audit_repository_cleanliness.py
python tools/audit_baseline_proposal_comparison.py
python tools/audit_baseline_dapr_strict.py --forward
python tools/audit_dapr_ablation.py
python tools/audit_dapr_hyperparameters.py
```

These checks ensure that:

- only DAPR is active as the proposal method;
- all baseline and DAPR configs share the same fair protocol;
- every baseline can build and run a small forward pass;
- DAPR ablation and hyperparameter configs only change their intended fields;
- no Python cache artifacts remain.

## Kaggle scripts

```bash
bash scripts/kaggle_baseline_proposal_etis_3seeds.sh
bash scripts/kaggle_dapr_ablation_etis_3seeds.sh
bash scripts/kaggle_dapr_hyperparameters_etis_3seeds.sh
```


## BSPC reviewer revision (2026-11-17)

A reviewer-focused code extension is documented in
[`docs/REVIEWER_CODE_MAPPING.md`](docs/REVIEWER_CODE_MAPPING.md) and
[`docs/REVIEWER_RUNBOOK_20261117.md`](docs/REVIEWER_RUNBOOK_20261117.md).
It preserves the original 2D direct-reconstruction BT U-Net, adds reproducibility
checks and reviewer experiments, and introduces a **separate** 3D research variant.
The project does **not** bundle original third-party GFUNet/AFDSeg model sources,
medical datasets, trained weights, or new measured benchmark results. Official
spectral models require independently pinned source, an explicit paper-to-code
audit, and genuine retraining before making empirical comparisons.


## Kaggle 11-hour autonomous experiment queue (1-based)

A ready-to-upload notebook is included as `BT_UNet_KAGGLE_11H.ipynb`.
It clones GitHub, downloads/prepares data, trains independent experiments,
checks evaluations, and creates a session ZIP. Resume by changing only
`FIRST_EXPERIMENT` to the number in `BT_UNET_NEXT_EXPERIMENT_ALL.txt`.
The deterministic queue contains 261 existing 2D experiments. **Official external
spectral model sources and real CT/MRI 3D experiments are NOT part of this
queue and require additional approved data/code.**

See `docs/KAGGLE_11H_ONE_BASED_RUN_GUIDE.md` for setup, data sources,
Kaggle GPU/Internet settings, session output and final archive merging.

## Active AUTO 324 revision
See [BT_UNET_AUTO_324_REVISION.md](BT_UNET_AUTO_324_REVISION.md) for the revised proposal reruns, nine datasets, LinkNet, archive reuse, and the current Kaggle notebook. The [AUTO 300 guide](BT_UNET_AUTO_300_REVISION.md) is historical and documents dataset sources and metric additions.
