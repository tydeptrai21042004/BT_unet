#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORK_ROOT="${WORK_ROOT:-/kaggle/working}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/bt_unet_data/etis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/bt_unet_outputs/baseline_proposal_etis_3seeds}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-${WORK_ROOT}/baseline_proposal_etis_3seeds_results.zip}"

python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q -r requirements.txt

python tools/audit_repository_cleanliness.py
python tools/audit_baseline_proposal_comparison.py --config-dir configs/fair
python tools/audit_baseline_dapr_strict.py --no-forward
python tools/audit_fairness.py

python scripts/run_baseline_proposal_comparison.py \
  --dataset etis \
  --data-root "${DATA_ROOT}" \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --config-dir configs/fair \
  --output-root "${OUTPUT_ROOT}" \
  --allow-insecure-download \
  --delete-checkpoints-after-eval \
  --no-run-tests \
  --no-run-runtime-baseline-audit

TABLE_DIR="${OUTPUT_ROOT}/results/tables"
test -s "${TABLE_DIR}/multi_seed_summary.csv"
test -s "${TABLE_DIR}/baseline_proposal_training_summary.csv"
test -s "${TABLE_DIR}/baseline_proposal_comparison.tex"
test -s "${TABLE_DIR}/baseline_proposal_deltas.csv"

rm -f "${RESULT_ARCHIVE}"
cd "${WORK_ROOT}"
zip -qr "${RESULT_ARCHIVE}" "$(realpath --relative-to="${WORK_ROOT}" "${OUTPUT_ROOT}/results")"
echo "Results archive: ${RESULT_ARCHIVE}"
