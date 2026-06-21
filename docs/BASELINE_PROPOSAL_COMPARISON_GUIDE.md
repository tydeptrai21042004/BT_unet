# Baseline versus DAPR comparison guide

The fair comparison contains:

- eleven controlled baselines;
- one proposal: **DAPR U-Net: Direct Amplitude--Phase Reconstruction U-Net**.

Run:

```bash
python scripts/run_baseline_proposal_comparison.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seeds 42,1,2
```

Before running, verify the scope with:

```bash
python tools/audit_baseline_proposal_comparison.py
python tools/audit_baseline_dapr_strict.py --no-forward
```
