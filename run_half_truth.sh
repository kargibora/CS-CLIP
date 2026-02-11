#!/bin/bash
# =============================================================================
# Half-Truth Vulnerability Experiment Runner
# =============================================================================
# This script runs the half-truth vulnerability analysis on CLIP models.
# It tests whether CLIP prefers longer incorrect captions over shorter correct ones.
#
# Usage:
#   ./run_half_truth.sh                                    # Run with defaults (OpenAI CLIP)
#   ./run_half_truth.sh --checkpoint_path /path/to/model.pt --checkpoint_type openclip
#   ./run_half_truth.sh --checkpoint_type huggingface --checkpoint_path "model-id"
#   ./run_half_truth.sh --dataset laion --tar_range 0-100  # Run on LAION dataset
#   ./run_half_truth.sh --help                             # Show this help
#
# Arguments:
#   --dataset           Dataset to use: "coco" or "laion" (default: coco)
#   --model_name        Model architecture (default: ViT-B-32)
#   --pretrained        Pretrained weights: "openai", "laion2b_s34b_b79k", etc.
#   --checkpoint_type   Type: "openclip", "huggingface", "tripletclip", "external", "dac", "clove"
#   --checkpoint_path   Path to checkpoint file or HuggingFace model ID
#   --config            Path to config file for checkpoint
#   --num_samples       Number of samples to generate, -1 for all (default: 1000)
#   --output_dir        Output directory for results
#   --seed              Random seed (default: 42)
#   --tar_range         Tar range for LAION dataset (default: 0-10)
#   --help              Show this help message
# =============================================================================

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION - Can be overridden by command line arguments
# -----------------------------------------------------------------------------

# Dataset: "coco" or "laion"
DATASET="coco"

# Model settings
MODEL_NAME="ViT-B-32"
PRETRAINED="openai"        # "openai", "laion2b_s34b_b79k", etc.
CHECKPOINT_TYPE="openclip" # "openclip", "huggingface", "tripletclip", "external", "dac", "clove"
CHECKPOINT_PATH=""         # Path to custom checkpoint (.pt file) or HuggingFace model ID
CONFIG=""                  # Path to config file for checkpoint

# Number of samples to generate (-1 for all)
NUM_SAMPLES=1000

# Output directory for results (will be auto-generated if not set)
OUTPUT_DIR=""

# Random seed for reproducibility
SEED=42

# -----------------------------------------------------------------------------
# COCO Dataset Settings (used when DATASET="coco")
# -----------------------------------------------------------------------------
COCO_JSON="swap_pos_json/coco_val/"
COCO_IMAGE_ROOT="."

# -----------------------------------------------------------------------------
# LAION Dataset Settings (used when DATASET="laion")
# -----------------------------------------------------------------------------
LAION_DATA_ROOT="/mnt/lustre/datasets/laion400m/laion400m-data"
LAION_JSON_ROOT="/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m-negatives-with-swaps/"
TAR_RANGE="0,10"  # Format: "start-end", e.g., "0-100" for tars 00000-00100

# -----------------------------------------------------------------------------
# Analysis Options
# -----------------------------------------------------------------------------
# Visualization
SAVE_EXAMPLES=true         # Save example visualizations
NUM_EXAMPLES=20            # Number of examples to save

# -----------------------------------------------------------------------------
# SLURM Settings (for cluster submission)
# -----------------------------------------------------------------------------
USE_SLURM=false
SLURM_PARTITION="gpu"
SLURM_GPUS=1
SLURM_TIME="04:00:00"
SLURM_MEM="32G"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

show_help() {
    head -28 "$0" | tail -26
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_name|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED="$2"
            shift 2
            ;;
        --checkpoint_type|--ckpt_type)
            CHECKPOINT_TYPE="$2"
            shift 2
            ;;
        --checkpoint_path|--checkpoint|--ckpt)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --config|--cfg)
            CONFIG="$2"
            shift 2
            ;;
        --num_samples|--samples|-n)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir|--output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --tar_range)
            TAR_RANGE="$2"
            shift 2
            ;;
        --coco_json)
            COCO_JSON="$2"
            shift 2
            ;;
        --image_root)
            COCO_IMAGE_ROOT="$2"
            shift 2
            ;;
        --laion_data_root)
            LAION_DATA_ROOT="$2"
            shift 2
            ;;
        --laion_json_root)
            LAION_JSON_ROOT="$2"
            shift 2
            ;;
        --slurm)
            USE_SLURM=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# AUTO-GENERATE OUTPUT DIRECTORY IF NOT SET
# =============================================================================

if [ -z "$OUTPUT_DIR" ]; then
    # Generate output dir based on model and dataset
    if [ -n "$CHECKPOINT_PATH" ]; then
        # Extract checkpoint name without path and extension
        CKPT_NAME=$(basename "$CHECKPOINT_PATH" .pt)
        OUTPUT_DIR="./results/test_half_truth_${DATASET}_${CHECKPOINT_TYPE}_${CKPT_NAME}"
    else
        OUTPUT_DIR="./results/test_half_truth_${DATASET}_${PRETRAINED}"
    fi
fi

# =============================================================================
# BUILD AND RUN COMMAND
# =============================================================================

# Build the command
CMD="python experiments/half_truth_vulnerability.py"

# Add dataset
CMD="$CMD --dataset $DATASET"

# Add model settings
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --checkpoint_type $CHECKPOINT_TYPE"

# Add pretrained or checkpoint path
if [ -n "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --checkpoint_path $CHECKPOINT_PATH"
fi
CMD="$CMD --pretrained $PRETRAINED"

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

# Add common settings
CMD="$CMD --num_samples $NUM_SAMPLES"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --seed $SEED"

# Add visualization settings
if [ "$SAVE_EXAMPLES" = true ]; then
    CMD="$CMD --save_examples"
    CMD="$CMD --num_examples $NUM_EXAMPLES"
fi

# Add dataset-specific settings
if [ "$DATASET" = "coco" ]; then
    CMD="$CMD --json_folder $COCO_JSON"
    CMD="$CMD --image_root $COCO_IMAGE_ROOT"
elif [ "$DATASET" = "laion" ]; then
    CMD="$CMD --laion_data_root $LAION_DATA_ROOT"
    CMD="$CMD --laion_json_root $LAION_JSON_ROOT"
    CMD="$CMD --tar_range $TAR_RANGE"
fi

# Print configuration
echo "============================================================"
echo "Half-Truth Vulnerability Experiment"
echo "============================================================"
echo "Dataset:            $DATASET"
echo "Model:              $MODEL_NAME"
echo "Checkpoint type:    $CHECKPOINT_TYPE"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint path:    $CHECKPOINT_PATH"
fi
echo "Pretrained:         $PRETRAINED"
if [ -n "$CONFIG" ]; then
    echo "Config:             $CONFIG"
fi
echo "Num samples:        $NUM_SAMPLES"
echo "Output dir:         $OUTPUT_DIR"
echo "Seed:               $SEED"
echo ""
if [ "$DATASET" = "coco" ]; then
    echo "COCO settings:"
    echo "  JSON:             $COCO_JSON"
    echo "  Image root:       $COCO_IMAGE_ROOT"
elif [ "$DATASET" = "laion" ]; then
    echo "LAION settings:"
    echo "  Data root:        $LAION_DATA_ROOT"
    echo "  JSON root:        $LAION_JSON_ROOT"
    echo "  Tar range:        $TAR_RANGE"
fi
echo "============================================================"
echo ""
echo "Command: $CMD"
echo ""

# Run the experiment
if [ "$USE_SLURM" = true ]; then
    echo "Submitting to SLURM..."
    srun --partition=$SLURM_PARTITION \
         --gpus=$SLURM_GPUS \
         --time=$SLURM_TIME \
         --mem=$SLURM_MEM \
         $CMD
else
    echo "Running locally..."
    $CMD
fi

echo ""
echo "============================================================"
echo "Experiment complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
