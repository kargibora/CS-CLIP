#!/bin/bash
#SBATCH --job-name=coco_neg_gen
#SBATCH --partition=a100-galvani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/coco_neg_%A_%a.out
#SBATCH --error=logs/coco_neg_%A_%a.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ====================
# USER-CONFIGURABLE
# ====================

# COCO Dataset Settings
COCO_JSON="${COCO_JSON:-${SCRIPT_DIR}/datasets/COCO/dataset_coco.json}"        # Path to Karpathy split JSON
COCO_IMAGES_ROOT="${COCO_IMAGES_ROOT:-${SCRIPT_DIR}/datasets/COCO}"             # Root path for images
COCO_SPLIT="val"                                  # Split to process: train, val, test, restval, or empty for all

# Parallel Processing Settings
N_JOBS=8                                           # Should match --array=0-7 above
TOTAL_SAMPLES=0                                     # Will be auto-computed, or set manually if known

# LLM Settings
LLM_NAME="Qwen/Qwen3-14B-AWQ"
LLM_BATCH=256
# Structured generation settings
USE_RELATIONAL_EXTRACTION=true
N_NEG_PER_ENTITY=2
USE_RELATIONAL_NEGATIVES=true
GENERATE_POSITIVES=true

# Output paths
OUTPUT_BASE="neg_json/coco/coco-${COCO_SPLIT}"
POSITIVES_OUTPUT_BASE="pos_json/coco/coco-${COCO_SPLIT}"

# ====================
# ENVIRONMENT SETUP
# ====================

export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$SCRIPT_DIR"

if [ "${SKIP_CONDA:-0}" = "1" ]; then
    echo "Skipping conda activation (SKIP_CONDA=1)"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if [ -n "${CONDA_ENV_PATH:-}" ]; then
        conda activate "${CONDA_ENV_PATH}"
    elif [ -n "${CONDA_ENV_NAME:-}" ]; then
        conda activate "${CONDA_ENV_NAME}"
    else
        echo "No CONDA_ENV_PATH or CONDA_ENV_NAME set; using current shell environment."
    fi
else
    echo "Conda not found; using current shell environment."
fi

# Create log directory if needed
mkdir -p logs

# ====================
# COUNT TOTAL SAMPLES
# ====================

# Count samples in the specified split (or all if no split specified)
if [ "$COCO_SPLIT" != "" ]; then
    # Count samples matching the split
    TOTAL_SAMPLES=$(python -c "
import json
with open('$COCO_JSON') as f:
    data = json.load(f)
count = sum(len(img['sentences']) for img in data['images'] if img['split'] == '$COCO_SPLIT')
print(count)
")
else
    # Count all samples
    TOTAL_SAMPLES=$(python -c "
import json
with open('$COCO_JSON') as f:
    data = json.load(f)
count = sum(len(img['sentences']) for img in data['images'])
print(count)
")
fi

echo "Total samples in COCO ${COCO_SPLIT:-all} split: $TOTAL_SAMPLES"

# ====================
# DYNAMIC SUBSET LOGIC
# ====================

JOB_IDX=${SLURM_ARRAY_TASK_ID}

# Compute local range for each job
SAMPLES_PER_JOB=$(( (TOTAL_SAMPLES + N_JOBS - 1) / N_JOBS ))  # ceil division

# Start and end indices for this job (0-indexed)
START_IDX=$(( JOB_IDX * SAMPLES_PER_JOB ))
END_IDX=$(( START_IDX + SAMPLES_PER_JOB ))

# Clip to not exceed total samples
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then
    END_IDX=$TOTAL_SAMPLES
fi

# Skip if this job has no work (edge case for last jobs)
if [ $START_IDX -ge $TOTAL_SAMPLES ]; then
    echo "Job $JOB_IDX has no samples to process (start=$START_IDX >= total=$TOTAL_SAMPLES)"
    exit 0
fi

OUTPUT_FILE="${OUTPUT_BASE}-${START_IDX}-${END_IDX}.json"
POSITIVES_OUTPUT_FILE="${POSITIVES_OUTPUT_BASE}-${START_IDX}-${END_IDX}.json"

# Create output directories if they don't exist
mkdir -p $(dirname "$OUTPUT_FILE")
mkdir -p $(dirname "$POSITIVES_OUTPUT_FILE")

echo "========================================"
echo "COCO Negative Generation - Job $JOB_IDX"
echo "========================================"
echo "COCO JSON: $COCO_JSON"
echo "Images root: $COCO_IMAGES_ROOT"
echo "Split: ${COCO_SPLIT:-all}"
echo "Sample range: $START_IDX to $END_IDX ($(($END_IDX - $START_IDX)) samples)"
echo "Output files:"
echo "  Negatives: $OUTPUT_FILE"
echo "  Positives: $POSITIVES_OUTPUT_FILE"
echo ""

# ====================
# BUILD COMMAND
# ====================

CMD="python cli.py \
    --coco_karpathy \"$COCO_JSON\" \
    --coco_images_root \"$COCO_IMAGES_ROOT\" \
    --output \"$OUTPUT_FILE\" \
    --llm_name \"$LLM_NAME\" \
    --llm_batch \"$LLM_BATCH\" \
    --subset $START_IDX $END_IDX"

# Add split filter if specified
if [ "$COCO_SPLIT" != "" ]; then
    CMD="$CMD --coco_split $COCO_SPLIT"
fi

echo "Pipeline: Structured COCO extraction and generation"

if [ "$USE_RELATIONAL_EXTRACTION" = true ]; then
    CMD="$CMD --use_relational_extraction"
    CMD="$CMD --n_neg_per_entity $N_NEG_PER_ENTITY"
fi

if [ "$USE_RELATIONAL_NEGATIVES" = true ]; then
    CMD="$CMD --use_relational_negatives"
fi

if [ "$GENERATE_POSITIVES" = true ]; then
    CMD="$CMD --positives_output \"$POSITIVES_OUTPUT_FILE\""
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""

# ====================
# EXECUTE
# ====================

eval $CMD

# ====================
# SUMMARY
# ====================

echo ""
echo "========================================"
echo "Job $JOB_IDX completed!"
echo "========================================"

if [ -f "$OUTPUT_FILE" ]; then
    NEG_COUNT=$(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))" 2>/dev/null || echo "unknown")
    echo "  Negatives: $OUTPUT_FILE ($NEG_COUNT samples)"
    
    # Show sample structure
    echo "  Sample structure:"
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
if data:
    sample = data[0]
    print(f'    Keys: {list(sample.keys())}')
    if 'negative_entities' in sample:
        print(f'    Negative types: {list(sample[\"negative_entities\"].keys()) if isinstance(sample[\"negative_entities\"], dict) else type(sample[\"negative_entities\"])}')
" 2>/dev/null || echo "  (unable to parse sample)"
else
    echo "  Negatives: $OUTPUT_FILE (file not found)"
fi

if [ "$GENERATE_POSITIVES" = true ]; then
    if [ -f "$POSITIVES_OUTPUT_FILE" ]; then
        POS_COUNT=$(python -c "import json; print(len(json.load(open('$POSITIVES_OUTPUT_FILE'))))" 2>/dev/null || echo "unknown")
        echo "  Positives: $POSITIVES_OUTPUT_FILE ($POS_COUNT samples)"
    else
        echo "  Positives: $POSITIVES_OUTPUT_FILE (file not found)"
    fi
fi
