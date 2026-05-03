#!/bin/bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
    echo "This script is configured with environment variables, not command-line flags."
    echo "Required: CHECKPOINT_PATH, CHECKPOINT_CONFIG"
    exit 1
fi

require_var() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Error: ${name} is required."
        exit 1
    fi
}

require_var CHECKPOINT_PATH
require_var CHECKPOINT_CONFIG

BASE_MODEL="${BASE_MODEL:-ViT-B/32}"
DATASETS="${DATASETS:-VG_Attribution}"
OUTPUT_CSV="${OUTPUT_CSV:-evaluation.csv}"
EVAL_NAME="${EVAL_NAME:-$(basename "${CHECKPOINT_PATH}" .pt)}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-}"
PYTHON_BIN="${PYTHON:-python3}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: CHECKPOINT_PATH does not exist: ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_CONFIG}" ]; then
    echo "Error: CHECKPOINT_CONFIG does not exist: ${CHECKPOINT_CONFIG}"
    exit 1
fi

if [ -n "${EVAL_DATA_ROOT}" ] && [ ! -d "${EVAL_DATA_ROOT}" ]; then
    echo "Error: EVAL_DATA_ROOT does not exist: ${EVAL_DATA_ROOT}"
    exit 1
fi

echo "=========================================="
echo "Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint:   ${CHECKPOINT_PATH}"
echo "Config:       ${CHECKPOINT_CONFIG}"
echo "Datasets:     ${DATASETS}"
if [ -n "${EVAL_DATA_ROOT}" ]; then
    echo "Data root:    ${EVAL_DATA_ROOT}"
fi
echo "Output CSV:   ${OUTPUT_CSV}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "=========================================="
echo ""

CMD=(
    "${PYTHON_BIN}" scripts/batch_evaluate_checkpoints.py
    --checkpoint_type local
    --checkpoint_path "${CHECKPOINT_PATH}"
    --base_model "${BASE_MODEL}"
    --is_finetuned true
    --base_config "${CHECKPOINT_CONFIG}"
    --name "${EVAL_NAME}"
    --csv_filename "${OUTPUT_CSV}"
    --output_dir "${OUTPUT_DIR}"
)

if [ -n "${EVAL_DATA_ROOT}" ]; then
    CMD+=(--dataset_root "${EVAL_DATA_ROOT}")
fi

CMD+=(--datasets ${DATASETS})

"${CMD[@]}"
