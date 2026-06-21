#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DAPR ABLATION
# Dataset: Kvasir-SEG
# Runs: 5 DAPR variants x 3 seeds = 15 runs
# ============================================================

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="-p no:cacheprovider"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORK_ROOT="${WORK_ROOT:-/kaggle/working}"
REPO_URL="https://github.com/tydeptrai21042004/BT_unet.git"
RUN_TAG="ablation_kvasir_seg"
REPO_DIR="${WORK_ROOT}/BT_unet_${RUN_TAG}"

DATASET="kvasir_seg"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/bt_unet_data/${RUN_TAG}}"
SEEDS="${SEEDS:-42,1,2}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_STRATEGY="random"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/bt_unet_outputs/${RUN_TAG}}"
RESULT_ARCHIVE="${WORK_ROOT}/${RUN_TAG}_results.zip"

MODELS="dapr_residual_control,dapr_no_global_phase,dapr_no_global_amplitude,dapr_no_global_channel_mix,dapr_unet"
CONFIG_DIR="configs/dapr_ablation"

SOURCE_DIR="${SOURCE_DIR:-}"
ZIP_PATH="${ZIP_PATH:-}"
DOWNLOAD_URL="${DOWNLOAD_URL:-}"
DOWNLOAD_DST="${DOWNLOAD_DST:-}"

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

section "RUN SUMMARY"
echo "Repo:      ${REPO_URL}"
echo "Dataset:   ${DATASET}"
echo "Seeds:     ${SEEDS}"
echo "Runs:      15"
echo "Epochs:    ${EPOCHS}"
echo "Split:     ${SPLIT_STRATEGY}"
echo "Data root: ${DATA_ROOT}"
echo "Output:    ${OUTPUT_ROOT}"
echo "Archive:   ${RESULT_ARCHIVE}"
echo "WARNING: 15 runs on Kvasir-SEG may exceed 12 hours at 30 epochs."

section "CLONE REPOSITORY"
cd "${WORK_ROOT}"
rm -rf "${REPO_DIR}"
git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"
echo "Commit: $(git rev-parse HEAD)"

section "CLEAN CACHE"
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
if [[ -f scripts/clean_repository_artifacts.py ]]; then
    python scripts/clean_repository_artifacts.py --apply
fi

section "INSTALL DEPENDENCIES"
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q -r requirements.txt
python -m pip install -q pytest certifi

section "AUDIT CODEBASE"
python tools/audit_repository_cleanliness.py
python tools/audit_dapr_ablation.py --config-dir "${CONFIG_DIR}"

section "VERIFY CUDA"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: enable GPU in Kaggle or set DEVICE=cpu for a non-GPU smoke run.")
torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda")
print("CUDA test:", float((x @ x).mean().item()))
PY
nvidia-smi

section "DOWNLOAD OR PREPARE DATASET"
mkdir -p "${DATA_ROOT}"
DATA_ARGS=()
DOWNLOAD_ARGS=()

if [[ -n "${SOURCE_DIR}" ]]; then
    DATA_ARGS+=(--source-dir "${SOURCE_DIR}")
elif [[ -n "${ZIP_PATH}" ]]; then
    DATA_ARGS+=(--zip-path "${ZIP_PATH}")
elif [[ -n "${DOWNLOAD_URL}" ]]; then
    DATA_ARGS+=(--download-url "${DOWNLOAD_URL}")
    if [[ -n "${DOWNLOAD_DST}" ]]; then
        DATA_ARGS+=(--download-dst "${DOWNLOAD_DST}")
    fi
else
    DOWNLOAD_ARGS+=(--allow-insecure-download)
fi

python scripts/prepare_dataset.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --image-size "${IMAGE_SIZE}" \
    "${DATA_ARGS[@]}" \
    "${DOWNLOAD_ARGS[@]}"

python scripts/make_splits.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --image-size "${IMAGE_SIZE}" \
    --seed "${SPLIT_SEED}" \
    --strategy "${SPLIT_STRATEGY}"

section "RUN EXPERIMENTS"
rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -ra SEED_ARRAY <<< "${SEEDS}"
for RAW_SEED in "${SEED_ARRAY[@]}"; do
    SEED="$(echo "${RAW_SEED}" | xargs)"
    if [[ -z "${SEED}" ]]; then
        continue
    fi

    SEED_OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"
    echo
    echo "[SEED ${SEED}] output: ${SEED_OUTPUT}"

    python scripts/benchmark_all.py \
        --models "${MODELS}" \
        --config-dir "${CONFIG_DIR}" \
        --dataset "${DATASET}" \
        --data-root "${DATA_ROOT}" \
        --image-size "${IMAGE_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LEARNING_RATE}" \
        --device "${DEVICE}" \
        --num-workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --output-root "${SEED_OUTPUT}" \
        --skip-prepare \
        --skip-splits

    find "${SEED_OUTPUT}" -type d -name "checkpoints" -prune -exec rm -rf {} +
done

section "AGGREGATE AND REPORT"
TABLE_DIR="${OUTPUT_ROOT}/results/tables"
mkdir -p "${TABLE_DIR}"

python scripts/aggregate_seed_results.py \
    --output-root "${OUTPUT_ROOT}" \
    --seeds "${SEEDS}"

python scripts/aggregate_training_results.py \
    --output-root "${OUTPUT_ROOT}" \
    --output-path "${TABLE_DIR}/full_training_summary.csv" \
    --models "${MODELS}" \
    --seeds "${SEEDS}" \
    --dataset "${DATASET}" \
    --expected-batch-size "${BATCH_SIZE}"

python scripts/validate_ablation_results.py \
    --summary-path "${TABLE_DIR}/multi_seed_summary.csv" \
    --training-summary-path "${TABLE_DIR}/full_training_summary.csv" \
    --output-root "${OUTPUT_ROOT}" \
    --models "${MODELS}" \
    --seeds "${SEEDS}" \
    --dataset "${DATASET}"

python scripts/report_dapr_ablation.py \
    --summary-path "${TABLE_DIR}/multi_seed_summary.csv" \
    --latex-path "${TABLE_DIR}/dapr_ablation.tex" \
    --delta-path "${TABLE_DIR}/dapr_ablation_deltas.csv"

section "OUTPUT LIST"
echo "Expected outputs:"
echo "  ${TABLE_DIR}/multi_seed_summary.csv"
echo "  ${TABLE_DIR}/full_training_summary.csv"
echo "  ${TABLE_DIR}/dapr_ablation.tex"
echo "  ${TABLE_DIR}/dapr_ablation_deltas.csv"
echo "  ${RESULT_ARCHIVE}"

test -s "${TABLE_DIR}/multi_seed_summary.csv"
test -s "${TABLE_DIR}/full_training_summary.csv"
test -s "${TABLE_DIR}/dapr_ablation.tex"
test -s "${TABLE_DIR}/dapr_ablation_deltas.csv"

echo
echo "Generated files:"
find "${TABLE_DIR}" -maxdepth 1 -type f -print | sort

section "PRINT RESULTS"
cat "${TABLE_DIR}/multi_seed_summary.csv"
echo
cat "${TABLE_DIR}/full_training_summary.csv"
echo
cat "${TABLE_DIR}/dapr_ablation_deltas.csv"
echo
cat "${TABLE_DIR}/dapr_ablation.tex"

section "PACKAGE RESULTS"
rm -f "${RESULT_ARCHIVE}"
(
    cd "${OUTPUT_ROOT}"
    zip -qr "${RESULT_ARCHIVE}" results
)
echo "Packaged: ${RESULT_ARCHIVE}"

section "FINAL CACHE CLEANUP"
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

section "DONE"
echo "Results archive: ${RESULT_ARCHIVE}"
