#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BASELINE VS DAPR
# Dataset: ISBI 2012
# Runs: 12 models x 3 seeds = 36 runs
# ============================================================

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORK_ROOT="${WORK_ROOT:-/kaggle/working}"
REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/BT_unet.git}"
RUN_TAG="baseline_isbi2012"
REPO_DIR="${WORK_ROOT}/BT_unet_${RUN_TAG}"

DATASET="isbi2012"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/bt_unet_data/${RUN_TAG}}"
SEEDS="${SEEDS:-42,1,2}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/bt_unet_outputs/${RUN_TAG}_seeds_$(echo "${SEEDS}" | tr ',' '_')}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-${WORK_ROOT}/${RUN_TAG}_results.zip}"

SOURCE_DIR="${SOURCE_DIR:-}"
ZIP_PATH="${ZIP_PATH:-}"

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
echo "Runs:      36"
echo "Epochs:    ${EPOCHS}"
echo "Data root: ${DATA_ROOT}"
echo "Output:    ${OUTPUT_ROOT}"
echo "Archive:   ${RESULT_ARCHIVE}"
echo "NOTE: ISBI 2012 uses contiguous split because adjacent slices are correlated."

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

section "AUDIT CODEBASE BEFORE RUNTIME DATA"
python tools/audit_repository_cleanliness.py
python tools/audit_baseline_proposal_comparison.py --config-dir configs/fair
python tools/audit_baseline_dapr_strict.py --no-forward
python tools/audit_fairness.py

section "VERIFY CUDA"
python - <<'EOF_PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: enable GPU in Kaggle.")
torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda")
print("CUDA test:", float((x @ x).mean().item()))
EOF_PY
nvidia-smi

section "PREPARE DATASET"
DATA_ARGS=()
DOWNLOAD_ARGS=()
if [[ -n "${SOURCE_DIR}" ]]; then
    DATA_ARGS+=(--source-dir "${SOURCE_DIR}")
elif [[ -n "${ZIP_PATH}" ]]; then
    DATA_ARGS+=(--zip-path "${ZIP_PATH}")
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
    --strategy contiguous

section "RUN BASELINE VS DAPR"
rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

python scripts/run_baseline_proposal_comparison.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --device cuda \
    --num-workers "${NUM_WORKERS}" \
    --seeds "${SEEDS}" \
    --config-dir configs/fair \
    --output-root "${OUTPUT_ROOT}" \
    --delete-checkpoints-after-eval \
    --no-run-tests \
    --no-run-runtime-baseline-audit \
    "${DATA_ARGS[@]}" \
    "${DOWNLOAD_ARGS[@]}"

section "OUTPUT LIST"
TABLE_DIR="${OUTPUT_ROOT}/results/tables"
echo "Expected outputs:"
echo "  ${TABLE_DIR}/multi_seed_summary.csv"
echo "  ${TABLE_DIR}/baseline_proposal_training_summary.csv"
echo "  ${TABLE_DIR}/baseline_proposal_comparison.tex"
echo "  ${TABLE_DIR}/baseline_proposal_deltas.csv"
echo "  ${RESULT_ARCHIVE}"

test -s "${TABLE_DIR}/multi_seed_summary.csv"
test -s "${TABLE_DIR}/baseline_proposal_training_summary.csv"
test -s "${TABLE_DIR}/baseline_proposal_comparison.tex"
test -s "${TABLE_DIR}/baseline_proposal_deltas.csv"

find "${TABLE_DIR}" -maxdepth 1 -type f -print | sort

section "PRINT RESULTS"
cat "${TABLE_DIR}/multi_seed_summary.csv"
echo
cat "${TABLE_DIR}/baseline_proposal_training_summary.csv"
echo
cat "${TABLE_DIR}/baseline_proposal_deltas.csv"
echo
cat "${TABLE_DIR}/baseline_proposal_comparison.tex"

section "PACKAGE RESULTS"
rm -f "${RESULT_ARCHIVE}"
cd "${WORK_ROOT}"
zip -qr "${RESULT_ARCHIVE}" "$(realpath --relative-to="${WORK_ROOT}" "${OUTPUT_ROOT}/results")"
echo "Packaged: ${RESULT_ARCHIVE}"

section "DONE"
echo "Results archive: ${RESULT_ARCHIVE}"
