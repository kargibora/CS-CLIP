#!/bin/bash
# LAION Fine-Tuning (FT) Training Script
# Easy-to-use script for running LAION400M fine-tuning experiments

# ============================================================================
# ACTIVATE CONDA ENVIRONMENT
# ============================================================================
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /mnt/lustre/work/oh/owl336/.conda/py-311-pytorch

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi
echo "Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# Distributed training
NPROC_PER_NODE=8
MASTER_PORT=12346

# Base config
CONFIG_NAME="laion_ft"

# Head type (for FT, usually bimodal)
HEAD_TYPE="bimodal"

# Track which parameters were explicitly set by user
SET_FT_TEXT=false
SET_FT_IMAGE=false
SET_ALIGN_TEXT=false
SET_ALIGN_IMAGE=false
SET_BATCH_SIZE=false
SET_USE_AMP=false
SET_VAL_RATIO=false
SET_TAR_RANGE=false
SKIP_TAR_RANGE=false
SET_CACHE_FOLDER=false
SET_LEARNING_RATE=false
SET_WEIGHT_DECAY=false
SET_OPTIMIZER=false
SET_ENABLE_EVAL=false
SET_EVAL_DATASETS=false
SET_EVAL_CSV=false
SET_EXP_NAME=false

# Fine-tuning settings (only used if explicitly set)
FT_TEXT=""
FT_IMAGE=""

# Alignment settings (usually false for pure FT)
ALIGN_TEXT=false
ALIGN_IMAGE=false

# Training settings (only used if explicitly set)
BATCH_SIZE=256
USE_AMP=true
VAL_RATIO=0.01

# Dataset settings (only used if explicitly set)
TAR_RANGE="[0,256]"
CACHE_FOLDER=""

# Optimizer settings (only used if explicitly set)
OPTIMIZER=""
LEARNING_RATE=""
WEIGHT_DECAY=""

# Evaluation settings (only used if explicitly set)
ENABLE_EVAL=true
EVAL_DATASETS='["MMVP", "COLA", "SPEC_I2T", "VisMin", "VALSE", "BLA", "ColorFoil", "ColorSwap", "SugarCrepe_PP", "SugarCrepe", "VG_Attribution", "VG_Relation", "ControlledImages", "COCO_Counterfactuals", "Winoground", "NegBench", "COCO_Order", "Flickr30k_Order", "CLIPBench_wds_vtab-cifar10", "CLIPBench_wds_flickr8k", "CLIPBench_wds_imagenet1k", "CLIPBench_wds_vtab-caltech101", "CLIPBench_wds_mscoco_captions", "CLIPBench_wds_sun397", "CLIPBench_wds_imagenet_sketch", "CLIPBench_wds_imagenetv2", "CLIPBench_wds_imagenet-o"]'
# EVAL_DATASETS='["SugarCrepe_PP"]'
EVAL_CSV="oct7_ft_text_5e-6.csv"

# Experiment name (only used if explicitly set)
EXP_NAME="ft_text_1p_"

# Auto-confirm (skip interactive prompt)
AUTO_CONFIRM=true

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================

# Display help
show_help() {
    cat << EOF
Usage: ${0##*/} [OPTIONS]

LAION Fine-Tuning (FT) Training Script - Easy configuration for fine-tuning experiments

Note: If you don't provide an option, the default from the config file will be used.

OPTIONS:
    -h, --help              Show this help message
    
    # Distributed Settings
    --gpus N                Number of GPUs (default: $NPROC_PER_NODE)
    --port N                Master port (default: $MASTER_PORT)
    
    # Model & Head
    --config NAME           Config name (default: $CONFIG_NAME)
    --head TYPE             Head type: bimodal|cross_modal_mlp|linear (default: $HEAD_TYPE)
    --model ARCH            Model architecture: vit_b32|vit_l14 (default: from config)
    
    # Fine-Tuning Control
    --ft-text [BOOL]        Fine-tune text encoder (default: from config)
    --ft-image [BOOL]       Fine-tune image encoder (default: from config)
    --ft-both               Fine-tune both encoders (shortcut)
    --ft-none               Fine-tune neither (freeze both)
    
    # Alignment (usually disabled for pure FT)
    --align-text [BOOL]     Align text (default: from config, usually false for FT)
    --align-image [BOOL]    Align image (default: from config, usually false for FT)
    --align-both            Align both image and text
    --align-none            Align neither
    
    # Training
    --batch-size N          Batch size per GPU (default: from config)
    --amp                   Use automatic mixed precision (default: from config)
    --no-amp                Disable AMP
    --val-ratio R           Validation ratio (default: from config)
    --epochs N              Number of epochs (default: from config)
    
    # Dataset
    --tar-range RANGE       Tar range e.g., [0,256] or [0,41407] for full (default: from config)
    --cache-folder PATH     Cache folder (default: from config)
    
    # Optimizer
    --optimizer NAME        Optimizer: adamw_ft|adamw|sgd (default: from config)
    --lr RATE               Learning rate (default: from config, typical: 5e-6 for FT)
    --wd DECAY              Weight decay (default: from config)
    
    # Evaluation
    --eval                  Enable evaluation
    --no-eval               Disable evaluation
    --eval-datasets DATASETS Datasets to evaluate (default: from config)
    --eval-csv PATH         CSV output path (default: from config)
    
    # Experiment
    --name NAME             Experiment name (default: from config)
    
    # Advanced
    --extra-args "ARGS"     Additional Hydra arguments (deprecated: use key=value instead)
    
    # Direct Hydra Overrides (No -- prefix needed)
    key=value               Any Hydra config override in key=value format
                            Examples: model.learnable_alphas=true
                                     optimizer.scheduler=cosine
                                     training.gradient_checkpointing=true

EXAMPLES:
    # Fine-tune both encoders with all config defaults
    ${0##*/} --ft-both
    
    # Fine-tune text encoder only
    ${0##*/} --ft-text true --ft-image false
    
    # Fine-tune with custom learning rate
    ${0##*/} --ft-both --lr 1e-5
    
    # Full LAION fine-tuning with 8 GPUs
    ${0##*/} --gpus 8 --ft-both --tar-range [0,41407] --lr 5e-6 --wd 1e-2
    
    # Fine-tune image encoder only with custom settings
    ${0##*/} --ft-image true --ft-text false --lr 1e-6 --amp
    
    # Mix FT with alignment (hybrid approach)
    ${0##*/} --ft-text true --align-image true --lr 5e-6
    
    # Use ViT-L/14 for fine-tuning
    ${0##*/} --model vit_l14 --ft-both --lr 1e-6
    
    # Custom optimizer and scheduler
    ${0##*/} --ft-both --optimizer adamw --lr 5e-6 optimizer.scheduler=cosine
    
    # Quick test with small dataset
    ${0##*/} --ft-both --tar-range [0,16] --no-eval --epochs 5

EOF
}

# Parse arguments
EXTRA_ARGS=""
HYDRA_OVERRIDES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --gpus)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --head)
            HEAD_TYPE="$2"
            shift 2
            ;;
        --model)
            EXTRA_ARGS="$EXTRA_ARGS model=$2"
            shift 2
            ;;
        --ft-text)
            # Smart boolean handling - can be used as flag or with value
            if [[ -n "$2" && "$2" != --* && "$2" != *=* ]]; then
                FT_TEXT="$2"
                shift 2
            else
                FT_TEXT=true
                shift
            fi
            SET_FT_TEXT=true
            ;;
        --ft-image)
            # Smart boolean handling
            if [[ -n "$2" && "$2" != --* && "$2" != *=* ]]; then
                FT_IMAGE="$2"
                shift 2
            else
                FT_IMAGE=true
                shift
            fi
            SET_FT_IMAGE=true
            ;;
        --ft-both)
            FT_TEXT=true
            FT_IMAGE=true
            SET_FT_TEXT=true
            SET_FT_IMAGE=true
            shift
            ;;
        --ft-none)
            FT_TEXT=false
            FT_IMAGE=false
            SET_FT_TEXT=true
            SET_FT_IMAGE=true
            shift
            ;;
        --align-text)
            # Smart boolean handling
            if [[ -n "$2" && "$2" != --* && "$2" != *=* ]]; then
                ALIGN_TEXT="$2"
                shift 2
            else
                ALIGN_TEXT=true
                shift
            fi
            SET_ALIGN_TEXT=true
            ;;
        --align-image)
            # Smart boolean handling
            if [[ -n "$2" && "$2" != --* && "$2" != *=* ]]; then
                ALIGN_IMAGE="$2"
                shift 2
            else
                ALIGN_IMAGE=true
                shift
            fi
            SET_ALIGN_IMAGE=true
            ;;
        --align-both)
            ALIGN_TEXT=true
            ALIGN_IMAGE=true
            SET_ALIGN_TEXT=true
            SET_ALIGN_IMAGE=true
            shift
            ;;
        --align-none)
            ALIGN_TEXT=false
            ALIGN_IMAGE=false
            SET_ALIGN_TEXT=true
            SET_ALIGN_IMAGE=true
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            SET_BATCH_SIZE=true
            shift 2
            ;;
        --amp)
            USE_AMP=true
            SET_USE_AMP=true
            shift
            ;;
        --no-amp)
            USE_AMP=false
            SET_USE_AMP=true
            shift
            ;;
        --val-ratio)
            VAL_RATIO="$2"
            SET_VAL_RATIO=true
            shift 2
            ;;
        --epochs)
            EXTRA_ARGS="$EXTRA_ARGS training.epochs=$2"
            shift 2
            ;;
        --tar-range)
            TAR_RANGE="$2"
            SET_TAR_RANGE=true
            shift 2
            ;;
        --skip-tar-range)
            SKIP_TAR_RANGE=true
            shift
            ;;
        --cache-folder)
            CACHE_FOLDER="$2"
            SET_CACHE_FOLDER=true
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            SET_OPTIMIZER=true
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            SET_LEARNING_RATE=true
            shift 2
            ;;
        --wd)
            WEIGHT_DECAY="$2"
            SET_WEIGHT_DECAY=true
            shift 2
            ;;
        --eval)
            ENABLE_EVAL=true
            SET_ENABLE_EVAL=true
            shift
            ;;
        --no-eval)
            ENABLE_EVAL=false
            SET_ENABLE_EVAL=true
            shift
            ;;
        --eval-datasets)
            EVAL_DATASETS="$2"
            SET_EVAL_DATASETS=true
            shift 2
            ;;
        --eval-csv)
            EVAL_CSV="$2"
            SET_EVAL_CSV=true
            shift 2
            ;;
        --name)
            EXP_NAME="$2"
            SET_EXP_NAME=true
            shift 2
            ;;
        -y|--yes)
            AUTO_CONFIRM=true
            shift
            ;;
        --extra-args)
            EXTRA_ARGS="$EXTRA_ARGS $2"
            shift 2
            ;;
        --*)
            # Unknown flag - show error
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            echo ""
            echo "Tip: For Hydra config overrides, use key=value format (without --)"
            echo "Example: ./train_laion_ft.sh model.learnable_alphas=true optimizer.scheduler=cosine"
            exit 1
            ;;
        *=*)
            # Direct Hydra override (key=value format)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES $1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            echo ""
            echo "Tip: For Hydra config overrides, use key=value format"
            echo "Example: ./train_laion_ft.sh model.learnable_alphas=true optimizer.scheduler=cosine"
            exit 1
            ;;
    esac
done

# ============================================================================
# BUILD COMMAND
# ============================================================================

# Start building the command
CMD="torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
  align.py --config-name=$CONFIG_NAME \
  head=$HEAD_TYPE"

# Only add overrides for parameters that were explicitly set OR have default values in script

# Fine-tuning settings
if [ "$SET_FT_TEXT" = true ] || [ -n "$FT_TEXT" ]; then
    [ -n "$FT_TEXT" ] && CMD="$CMD alignment.ft_text=$FT_TEXT"
fi

if [ "$SET_FT_IMAGE" = true ] || [ -n "$FT_IMAGE" ]; then
    [ -n "$FT_IMAGE" ] && CMD="$CMD alignment.ft_image=$FT_IMAGE"
fi

# Alignment settings
if [ "$SET_ALIGN_TEXT" = true ] || [ -n "$ALIGN_TEXT" ]; then
    [ -n "$ALIGN_TEXT" ] && CMD="$CMD alignment.align_text=$ALIGN_TEXT"
fi

if [ "$SET_ALIGN_IMAGE" = true ] || [ -n "$ALIGN_IMAGE" ]; then
    [ -n "$ALIGN_IMAGE" ] && CMD="$CMD alignment.align_image=$ALIGN_IMAGE"
fi

# Training settings
if [ "$SET_USE_AMP" = true ] || [ -n "$USE_AMP" ]; then
    [ -n "$USE_AMP" ] && CMD="$CMD training.use_amp=$USE_AMP"
fi

if [ "$SET_CACHE_FOLDER" = true ] || [ -n "$CACHE_FOLDER" ]; then
    [ -n "$CACHE_FOLDER" ] && CMD="$CMD cache.cache_folder=$CACHE_FOLDER"
fi

if [ "$SET_VAL_RATIO" = true ] || [ -n "$VAL_RATIO" ]; then
    [ -n "$VAL_RATIO" ] && CMD="$CMD "dataset".val_ratio=$VAL_RATIO"
fi

if [ "$SET_BATCH_SIZE" = true ] || [ -n "$BATCH_SIZE" ]; then
    [ -n "$BATCH_SIZE" ] && CMD="$CMD training.batch_size=$BATCH_SIZE"
fi

# Optimizer settings
if [ "$SET_OPTIMIZER" = true ] || [ -n "$OPTIMIZER" ]; then
    [ -n "$OPTIMIZER" ] && CMD="$CMD optimizer=$OPTIMIZER"
fi

if [ "$SET_LEARNING_RATE" = true ] || [ -n "$LEARNING_RATE" ]; then
    [ -n "$LEARNING_RATE" ] && CMD="$CMD optimizer.learning_rate=$LEARNING_RATE"
fi

if [ "$SET_WEIGHT_DECAY" = true ] || [ -n "$WEIGHT_DECAY" ]; then
    [ -n "$WEIGHT_DECAY" ] && CMD="$CMD optimizer.optimizer_kwargs.weight_decay=$WEIGHT_DECAY"
fi

# Evaluation settings
if [ "$SET_ENABLE_EVAL" = true ] || [ -n "$ENABLE_EVAL" ]; then
    [ -n "$ENABLE_EVAL" ] && CMD="$CMD evaluation.enable_dataset_eval=$ENABLE_EVAL"
fi

if [ "$SET_EVAL_DATASETS" = true ] || [ -n "$EVAL_DATASETS" ]; then
    [ -n "$EVAL_DATASETS" ] && CMD="$CMD 'evaluation.dataset_eval_datasets=$EVAL_DATASETS'"
fi

if [ "$SET_EVAL_CSV" = true ] || [ -n "$EVAL_CSV" ]; then
    [ -n "$EVAL_CSV" ] && CMD="$CMD evaluation.dataset_eval_csv_path=\"$EVAL_CSV\""
fi

# Dataset settings
if [ "$SKIP_TAR_RANGE" = false ]; then
    if [ "$SET_TAR_RANGE" = true ] || [ -n "$TAR_RANGE" ]; then
        [ -n "$TAR_RANGE" ] && CMD="$CMD dataset.dataset_kwargs.tar_range=$TAR_RANGE"
    fi
fi

# Experiment name
if [ "$SET_EXP_NAME" = true ] || [ -n "$EXP_NAME" ]; then
    [ -n "$EXP_NAME" ] && CMD="$CMD training.name=\"$EXP_NAME\""
fi

# Add any extra arguments
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Add any direct Hydra overrides
if [ -n "$HYDRA_OVERRIDES" ]; then
    CMD="$CMD $HYDRA_OVERRIDES"
fi

# ============================================================================
# DISPLAY CONFIGURATION & RUN
# ============================================================================

echo "============================================================================"
echo "LAION Fine-Tuning (FT) Training Configuration"
echo "============================================================================"
echo "GPUs:              $NPROC_PER_NODE"
echo "Master Port:       $MASTER_PORT"
echo "Config:            $CONFIG_NAME"
echo "Head Type:         $HEAD_TYPE"
echo ""
echo "Fine-Tuning:"
echo "  FT Text:           ${FT_TEXT:-'<from config>'}"
echo "  FT Image:          ${FT_IMAGE:-'<from config>'}"
echo ""
echo "Alignment (usually disabled for pure FT):"
echo "  Align Text:        ${ALIGN_TEXT:-'<from config>'}"
echo "  Align Image:       ${ALIGN_IMAGE:-'<from config>'}"
echo ""
echo "Training Settings:"
echo "  Batch Size:        ${BATCH_SIZE:-'<from config>'}"
echo "  Use AMP:           ${USE_AMP:-'<from config>'}"
echo "  Val Ratio:         ${VAL_RATIO:-'<from config>'}"
echo ""
echo "Optimizer:"
echo "  Optimizer:         ${OPTIMIZER:-'<from config>'}"
echo "  Learning Rate:     ${LEARNING_RATE:-'<from config>'}"
echo "  Weight Decay:      ${WEIGHT_DECAY:-'<from config>'}"
echo ""
echo "Dataset:"
echo "  Tar Range:         ${TAR_RANGE:-'<from config>'}"
echo "  Cache Folder:      ${CACHE_FOLDER:-'<from config>'}"
echo ""
echo "Evaluation:"
echo "  Enable Eval:       ${ENABLE_EVAL:-'<from config>'}"
echo "  Eval Datasets:     ${EVAL_DATASETS:-'<from config>'}"
echo "  Eval CSV:          ${EVAL_CSV:-'<from config>'}"
echo ""
echo "Experiment:"
echo "  Name:              ${EXP_NAME:-'<from config>'}"
if [ -n "$EXTRA_ARGS" ]; then
    echo ""
    echo "Extra Args:        $EXTRA_ARGS"
fi
if [ -n "$HYDRA_OVERRIDES" ]; then
    echo "Hydra Overrides:   $HYDRA_OVERRIDES"
fi
echo "============================================================================"
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "============================================================================"
echo ""

# Ask for confirmation (skip if AUTO_CONFIRM is set)
if [ "$AUTO_CONFIRM" = true ]; then
    echo "Starting fine-tuning (auto-confirmed)..."
    eval $CMD
else
    read -p "Start fine-tuning? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting fine-tuning..."
        eval $CMD
    else
        echo "Fine-tuning cancelled."
        exit 0
    fi
fi
