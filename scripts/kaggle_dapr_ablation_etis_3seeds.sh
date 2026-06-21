#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q -r requirements.txt
python -m pip install -q pytest certifi

python scripts/run_dapr_ablation.py \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --config-dir configs/dapr_ablation \
  --output-root outputs_dapr_ablation/etis \
  --allow-insecure-download \
  --delete-checkpoints-after-eval
