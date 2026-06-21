#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-kvasir_seg}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-42,1,2}"

FAIR_MODELS="unet,unetpp,attention_unet,pranet,acsnet,hardnet_mseg,cfanet,polyp_pvt,caranet,hsnet,resunetpp,plain_fourier_unet,apdr_fourier_unet"
FAITHFUL_MODELS="unet,unetpp,attention_unet,pranet,acsnet,hardnet_mseg,cfanet,polyp_pvt,caranet,hsnet,resunetpp"

case "${1:-help}" in
  install)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  prepare)
    shift
    "$PYTHON_BIN" scripts/prepare_dataset.py --dataset "$DATASET" --data-root "$DATA_ROOT" "$@"
    ;;
  fair)
    "$PYTHON_BIN" scripts/benchmark_all.py \
      --models "$FAIR_MODELS" \
      --config-dir configs/fair \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --device "$DEVICE" \
      --output-root "$OUTPUT_ROOT"
    ;;
  faithful)
    "$PYTHON_BIN" scripts/benchmark_all.py \
      --models "$FAITHFUL_MODELS" \
      --config-dir configs/official_faithful \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --device "$DEVICE" \
      --output-root "$OUTPUT_ROOT"
    ;;
  ablation)
    "$PYTHON_BIN" scripts/run_apdr_ablation.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --device "$DEVICE" \
      --seed "$SEED" \
      --output-root "$OUTPUT_ROOT"
    ;;
  two-proposal)
    "$PYTHON_BIN" scripts/run_two_proposal_comparison.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --device "$DEVICE" \
      --seeds "$SEEDS" \
      --output-root "$OUTPUT_ROOT"
    ;;
  baseline-proposal)
    "$PYTHON_BIN" scripts/run_baseline_proposal_comparison.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --device "$DEVICE" \
      --seeds "$SEEDS" \
      --output-root "$OUTPUT_ROOT"
    ;;
  full-ablation)
    "$PYTHON_BIN" scripts/run_full_component_ablation.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --device "$DEVICE" \
      --seeds "$SEEDS" \
      --output-root "$OUTPUT_ROOT"
    ;;
  audit)
    "$PYTHON_BIN" tools/audit_baseline_proposal_comparison.py
    "$PYTHON_BIN" tools/audit_baseline_implementations.py
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
    echo "Usage: bash run.sh {install|prepare|fair|faithful|ablation|two-proposal|baseline-proposal|full-ablation|audit|clean|smoke|test}"
    ;;
esac
