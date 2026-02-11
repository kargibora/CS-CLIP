#!/bin/bash
# =============================================================================
# Batch Triplet Embeddings Computation Runner
# =============================================================================
# This script reads checkpoints from eval_checkpoints.yaml and computes
# triplet embeddings for each one (for geometry analysis).
#
# Usage:
#   ./run_triplet_embeddings_batch.sh                    # Run all checkpoints
#   ./run_triplet_embeddings_batch.sh --dry-run          # Show commands without running
#   ./run_triplet_embeddings_batch.sh --config other.yaml # Use different config
#   ./run_triplet_embeddings_batch.sh --filter "CLIP"    # Run only checkpoints matching filter
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------

CONFIG_FILE="configs/eval_checkpoints.yaml"
JSON_PATH="swap_pos_json/coco_val/"
IMAGE_ROOT="."
OUTPUT_DIR="triplet_embeddings"
NUM_SAMPLES=5000
SEED=42
DRY_RUN=false
FILTER=""

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --json-path)
            JSON_PATH="$2"
            shift 2
            ;;
        --image-root)
            IMAGE_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE       Config file (default: configs/eval_checkpoints.yaml)"
            echo "  --json-path PATH    Path to JSON data (default: swap_pos_json/coco_val/)"
            echo "  --image-root PATH   Image root directory (default: .)"
            echo "  --output-dir DIR    Output directory (default: triplet_embeddings)"
            echo "  --num-samples N     Number of samples (default: 5000)"
            echo "  --seed N            Random seed (default: 42)"
            echo "  --dry-run           Show commands without running"
            echo "  --filter PATTERN    Only run checkpoints matching pattern"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# CHECK DEPENDENCIES
# =============================================================================

if ! command -v python &> /dev/null; then
    echo "Error: python not found"
    exit 1
fi

if ! python -c "import yaml" &> /dev/null; then
    echo "Error: PyYAML not installed. Run: pip install pyyaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# PARSE YAML AND RUN EMBEDDINGS
# =============================================================================

echo "============================================================"
echo "Batch Triplet Embeddings Computation"
echo "============================================================"
echo "Config file: $CONFIG_FILE"
echo "JSON path: $JSON_PATH"
echo "Image root: $IMAGE_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Num samples: $NUM_SAMPLES"
echo "Seed: $SEED"
if [ -n "$FILTER" ]; then
    echo "Filter: $FILTER"
fi
echo "============================================================"
echo ""

# Parse checkpoints from YAML
CHECKPOINTS=$(python -c "
import yaml
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

checkpoints = config.get('checkpoints', [])
for cp in checkpoints:
    if cp is None:
        continue
    name = cp.get('name', '')
    csv_filename = cp.get('csv_filename', '')
    checkpoint_type = cp.get('checkpoint_type', 'openclip')
    checkpoint_path = cp.get('checkpoint_path', '')
    base_model = cp.get('base_model', 'ViT-B/32')
    force_openclip = cp.get('force_openclip', False)
    pretrained = cp.get('pretrained', 'openai')
    clove_weight = cp.get('clove_weight', 0.6)
    
    # Skip if commented out (None values)
    if not name or not checkpoint_path:
        continue
    
    # Apply filter if specified
    filter_pattern = '$FILTER'
    if filter_pattern and filter_pattern.lower() not in name.lower():
        continue
    
    # Output as tab-separated values
    print(f'{name}\t{csv_filename}\t{checkpoint_type}\t{checkpoint_path}\t{base_model}\t{force_openclip}\t{pretrained}\t{clove_weight}')
")

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found in config file (or none match filter)"
    exit 0
fi

# Count checkpoints
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -l)
echo "Found $NUM_CHECKPOINTS checkpoint(s) to process"
echo ""

# Process each checkpoint
COUNTER=0
echo "$CHECKPOINTS" | while IFS=$'\t' read -r NAME CSV_FILENAME CHECKPOINT_TYPE CHECKPOINT_PATH BASE_MODEL FORCE_OPENCLIP PRETRAINED CLOVE_WEIGHT; do
    COUNTER=$((COUNTER + 1))
    
    echo "============================================================"
    echo "[$COUNTER/$NUM_CHECKPOINTS] Processing: $NAME"
    echo "============================================================"
    
    # Generate output filename from csv_filename (remove .csv, add _triplet_embeddings.npz)
    OUTPUT_NAME=$(echo "$CSV_FILENAME" | sed 's/\.csv$//')
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_triplet_embeddings.npz"
    
    # Skip if output already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  Output already exists: $OUTPUT_FILE"
        echo "  Skipping..."
        echo ""
        continue
    fi
    
    # Build command
    CMD="python experiments/compute_triplet_embeddings.py"
    CMD="$CMD --json_path \"$JSON_PATH\""
    CMD="$CMD --image_root \"$IMAGE_ROOT\""
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
    CMD="$CMD --num_samples $NUM_SAMPLES"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --checkpoint_type $CHECKPOINT_TYPE"
    CMD="$CMD --checkpoint_path \"$CHECKPOINT_PATH\""
    CMD="$CMD --model_name \"$BASE_MODEL\""
    CMD="$CMD --output_name \"$OUTPUT_NAME\""
    
    # Add force_openclip if true
    if [ "$FORCE_OPENCLIP" = "True" ] || [ "$FORCE_OPENCLIP" = "true" ]; then
        CMD="$CMD --force_openclip"
    fi
    
    # Add clove_weight for clove checkpoints
    if [ "$CHECKPOINT_TYPE" = "clove" ]; then
        CMD="$CMD --clove_weight $CLOVE_WEIGHT"
    fi
    
    echo "  Checkpoint type: $CHECKPOINT_TYPE"
    echo "  Checkpoint path: $CHECKPOINT_PATH"
    echo "  Base model: $BASE_MODEL"
    echo "  Output file: $OUTPUT_FILE"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would execute:"
        echo "  $CMD"
    else
        echo "  Executing..."
        eval $CMD
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Success: $OUTPUT_FILE"
        else
            echo "  ✗ Failed: $NAME"
        fi
    fi
    
    echo ""
done

echo "============================================================"
echo "Batch processing complete!"
echo "============================================================"
