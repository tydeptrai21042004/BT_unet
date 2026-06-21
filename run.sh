#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:--p no:cacheprovider}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-kvasir_seg}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-42,1,2}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.0003}"
NUM_WORKERS="${NUM_WORKERS:-2}"

FAIR_MODELS="unet,unetpp,attention_unet,pranet,acsnet,hardnet_mseg,cfanet,polyp_pvt,caranet,hsnet,resunetpp,dapr_unet"
FAITHFUL_MODELS="unet,unetpp,attention_unet,pranet,acsnet,hardnet_mseg,cfanet,polyp_pvt,caranet,hsnet,resunetpp"
DAPR_ABLATION_MODELS="dapr_residual_control,dapr_no_global_phase,dapr_no_global_amplitude,dapr_no_global_channel_mix,dapr_unet"
DAPR_HYPER_MODELS="dapr_unet,dapr_expansion_1_0,dapr_expansion_2_0,dapr_amp_scale_0_5,dapr_amp_scale_1_5,dapr_phase_pi_half,dapr_phase_2pi,dapr_dropout_0_0"

case "${1:-help}" in
  install)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  prepare)
    shift
    "$PYTHON_BIN" scripts/prepare_dataset.py --dataset "$DATASET" --data-root "$DATA_ROOT" --image-size "$IMAGE_SIZE" "$@"
    ;;
  split|splits)
    shift
    "$PYTHON_BIN" scripts/make_splits.py --dataset "$DATASET" --data-root "$DATA_ROOT" --image-size "$IMAGE_SIZE" "$@"
    ;;
  fair)
    "$PYTHON_BIN" scripts/benchmark_all.py \
      --models "$FAIR_MODELS" \
      --config-dir configs/fair \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --image-size "$IMAGE_SIZE" \
      --device "$DEVICE" \
      --output-root "$OUTPUT_ROOT"
    ;;
  faithful)
    "$PYTHON_BIN" scripts/benchmark_all.py \
      --models "$FAITHFUL_MODELS" \
      --config-dir configs/official_faithful \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --image-size "$IMAGE_SIZE" \
      --device "$DEVICE" \
      --output-root "$OUTPUT_ROOT"
    ;;
  baseline-proposal)
    "$PYTHON_BIN" scripts/run_baseline_proposal_comparison.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --num-workers "$NUM_WORKERS" \
      --seeds "$SEEDS" \
      --config-dir configs/fair \
      --output-root "$OUTPUT_ROOT"
    ;;
  ablation|dapr-ablation)
    "$PYTHON_BIN" scripts/run_dapr_ablation.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --num-workers "$NUM_WORKERS" \
      --seeds "$SEEDS" \
      --models "$DAPR_ABLATION_MODELS" \
      --config-dir configs/dapr_ablation \
      --output-root "$OUTPUT_ROOT"
    ;;
  hyperparameters|dapr-hyperparameters)
    "$PYTHON_BIN" scripts/run_dapr_hyperparameters.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --num-workers "$NUM_WORKERS" \
      --seeds "$SEEDS" \
      --models "$DAPR_HYPER_MODELS" \
      --config-dir configs/dapr_hyperparameters \
      --output-root "$OUTPUT_ROOT"
    ;;
  audit)
    "$PYTHON_BIN" tools/audit_baseline_proposal_comparison.py
    "$PYTHON_BIN" tools/audit_baseline_implementations.py --skip-runtime
    "$PYTHON_BIN" tools/audit_dapr_ablation.py
    "$PYTHON_BIN" tools/audit_dapr_hyperparameters.py
    "$PYTHON_BIN" tools/audit_repository_cleanliness.py
    ;;
  clean)
    "$PYTHON_BIN" scripts/clean_repository_artifacts.py --apply
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q -p no:cacheprovider
    ;;
  smoke)
    "$PYTHON_BIN" scripts/smoke_all_models.py --config-dir configs/fair
    ;;
  *)
    cat <<'USAGE'
Usage: bash run.sh {install|prepare|splits|fair|faithful|baseline-proposal|ablation|hyperparameters|audit|clean|smoke|test}

Useful environment variables:
  DATASET=kvasir_seg|cvc_clinicdb|etis|cvc_colondb|cvc_300|isbi2012|kvasir_instrument|hyper_kvasir_seg
  DATA_ROOT=data
  DEVICE=auto|cuda|cpu
  OUTPUT_ROOT=outputs
  SEEDS=42,1,2
  IMAGE_SIZE=352
  BATCH_SIZE=6
  EPOCHS=30
  LR=0.0003
  NUM_WORKERS=2
USAGE
    ;;
esac
