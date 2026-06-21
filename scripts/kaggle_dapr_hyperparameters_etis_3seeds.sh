#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="-p no:cacheprovider"

python scripts/run_dapr_hyperparameters.py \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --config-dir configs/dapr_hyperparameters \
  --output-root outputs_dapr_hyperparameters/etis \
  --allow-insecure-download \
  --delete-checkpoints-after-eval \
  --run-tests
