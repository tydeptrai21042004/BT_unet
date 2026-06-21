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
