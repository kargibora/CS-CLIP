#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt 0 ]; then
    echo "Use environment variables only."
    echo "Required: RUN_NAME, TRAIN_JSON_DIR, IMAGE_ROOT"
    exit 1
fi

require_var() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Error: ${name} is required."
        exit 1
    fi
}

require_var RUN_NAME
require_var TRAIN_JSON_DIR
require_var IMAGE_ROOT

if [ ! -d "${TRAIN_JSON_DIR}" ]; then
    echo "Error: TRAIN_JSON_DIR does not exist: ${TRAIN_JSON_DIR}"
    exit 1
fi

if [ ! -d "${IMAGE_ROOT}" ]; then
    echo "Error: IMAGE_ROOT does not exist: ${IMAGE_ROOT}"
    exit 1
fi

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-5}"
SAVE_EVERY_K_STEPS="${SAVE_EVERY_K_STEPS:-0}"
MASTER_PORT="${MASTER_PORT:-12346}"

if [ "${SAVE_EVERY_K_STEPS}" -gt 0 ]; then
    SAVE_EVERY_OVERRIDE="${SAVE_EVERY_K_STEPS}"
else
    SAVE_EVERY_OVERRIDE="null"
fi

echo "Run: ${RUN_NAME}"
echo "Train JSON: ${TRAIN_JSON_DIR}"
echo "Image root: ${IMAGE_ROOT}"
echo "GPUs: ${GPUS}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR: ${LR}"
echo "Epochs: ${EPOCHS}"
echo "Step checkpoint: ${SAVE_EVERY_OVERRIDE}"
echo ""

CMD=(
    torchrun
    "--nproc_per_node=${GPUS}"
    "--master_port=${MASTER_PORT}"
    "${SCRIPT_DIR}/align.py"
    "--config-name=coco_ft"
    "training.name=${RUN_NAME}"
    "training.batch_size=${BATCH_SIZE}"
    "training.epochs=${EPOCHS}"
    "training.save_every_k_steps=${SAVE_EVERY_OVERRIDE}"
    "optimizer.learning_rate=${LR}"
    "dataset.dataset_kwargs.json_folder=${TRAIN_JSON_DIR}"
    "dataset.dataset_kwargs.image_root=${IMAGE_ROOT}"
)

printf "Command:"
printf " %q" "${CMD[@]}"
printf "\n\n"

"${CMD[@]}"
