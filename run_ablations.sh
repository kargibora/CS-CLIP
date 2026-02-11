#!/bin/bash
# =============================================================================
# ABLATION STUDY LAUNCHER
# =============================================================================
# This script submits multiple SLURM jobs for systematic hyperparameter ablations.
# Each ablation is submitted as a separate job to maximize cluster utilization.
#
# Usage:
#   ./run_ablations.sh [--dry-run] [--ablation <name>] [--partition <partition>]
#
# Options:
#   --dry-run         Print commands without submitting jobs
#   --ablation <name> Run only specific ablation (lambdas, components, lr, negatives, paraphrase, all)
#   --partition <p>   Override partition (default: a100-galvani)
#   --time <time>     Override time limit (default: 48:00:00)
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# SLURM settings
PARTITION="a100-galvani"          # or "a100-fat-galvani"
TIME_LIMIT="48:00:00"             # 48 hours per job
GPUS_PER_JOB=8
CPUS_PER_GPU=4
MEM_PER_GPU="32G"

# Base experiment settings (your current best config)
BASE_MODEL="vit_b32_openai"
BASE_DATASET="coco"
BASE_BATCH_SIZE=128
BASE_LR="5e-6"
BASE_WD="1e-2"

# Loss weights
BASE_LAMBDA_FULL=1.0
BASE_LAMBDA_COMPONENTS=0.5
BASE_LAMBDA_ALIGNMENT=0.0
BASE_LAMBDA_RANK=0.0
BASE_LAMBDA_TEXT_CONTRASTIVE=0.0
BASE_LAMBDA_PARAPHRASE=0.0

# Loss types and modes
BASE_CONTRASTIVE_MODE="with_components_negatives"  # Loss mode
BASE_COMPONENT_LOSS_TYPE="negclip_hard"            # "clip", "negclip", or "negclip_hard"
BASE_ALIGNMENT_LOSS_TYPE="margin"                  # "cosine" or "margin"
BASE_ALIGNMENT_MARGIN=0.2
BASE_RANK_MARGIN=0.1
BASE_TEXT_CONTRASTIVE_MARGIN=0.2

# Component sampling
BASE_MAX_COMPONENTS=2
BASE_NUM_COMPONENT_CAPTIONS=2

# Structured sampling
BASE_USE_STRUCTURED_SAMPLING=true
BASE_STRUCTURED_RELATION_PROB=1.0
BASE_USE_CONTEXT_IN_COMPONENT_PAIRS=true
BASE_SAMPLE_REL_OR_COMP="mixed"
BASE_MIX_COMP_REL_PROB=0.0
BASE_RELATION_SAMPLE_PROB=0.5

# Negative sampling
BASE_NEG_REL_PROB=0.5
BASE_INPLACE_PROB=1.0
BASE_USE_SWAP_NEGATIVES=true
BASE_SWAP_PROB=1.0
BASE_USE_NEGATIVES_FULL=true

# Parse command line arguments
DRY_RUN=false
ABLATION_FILTER="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --ablation)
            ABLATION_FILTER="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run              Print commands without submitting"
            echo "  --ablation <name>      Run specific ablation:"
            echo "                         lambdas, components, lr, negatives, paraphrase, all"
            echo "  --partition <name>     SLURM partition (default: a100-galvani)"
            echo "  --time <HH:MM:SS>      Time limit per job (default: 48:00:00)"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

JOB_COUNT=0
SUBMITTED_JOBS=()

submit_job() {
    local JOB_NAME="$1"
    local EXTRA_ARGS="$2"
    
    JOB_COUNT=$((JOB_COUNT + 1))
    
    # Create SLURM script
    local SLURM_SCRIPT=$(mktemp /tmp/ablation_${JOB_NAME}_XXXXX.slurm)
    
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$((GPUS_PER_JOB * CPUS_PER_GPU))
#SBATCH --gres=gpu:${GPUS_PER_JOB}
#SBATCH --mem=$((GPUS_PER_JOB * 32))G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/ablation_%x_%j.out
#SBATCH --error=logs/ablation_%x_%j.err
#SBATCH --mail-type=END,FAIL

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "Job: \${SLURM_JOB_NAME}"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node: \${SLURM_NODELIST}"
echo "GPUs: ${GPUS_PER_JOB}"
echo "Start Time: \$(date)"
echo "=========================================="

# Change to working directory
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate /mnt/lustre/work/oh/owl336/.conda/py-311-pytorch

# Run training with environment variable overrides
# The EXTRA_ARGS contain VAR=value pairs that override train_structured.sh defaults
export ${EXTRA_ARGS}
./train_structured.sh \\
    --dataset ${BASE_DATASET} \\
    --gpus ${GPUS_PER_JOB} \\
    --tag "ablation_${JOB_NAME}"

echo "=========================================="
echo "End Time: \$(date)"
echo "=========================================="
EOF

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "=========================================="
        echo "[DRY RUN] Would submit job #${JOB_COUNT}: ${JOB_NAME}"
        echo "=========================================="
        echo "Extra args: ${EXTRA_ARGS}"
        echo "Script: ${SLURM_SCRIPT}"
        cat "$SLURM_SCRIPT"
    else
        echo ""
        echo "Submitting job #${JOB_COUNT}: ${JOB_NAME}"
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
        SUBMITTED_JOBS+=("${JOB_ID}:${JOB_NAME}")
        echo "  -> Job ID: ${JOB_ID}"
    fi
    
    rm -f "$SLURM_SCRIPT"
}

# =============================================================================
# ABLATION 1: Lambda Analysis (Rank Loss & Alignment Loss)
# =============================================================================

run_lambda_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 1: Lambda Analysis"
    echo "=============================================="
    
    # Rank Loss Lambda variations
    for LAMBDA_RANK in 0.1 0.2 0.3 0.5 1.0; do
        submit_job "rank_lambda_${LAMBDA_RANK}" \
            "LAMBDA_RANK=${LAMBDA_RANK}"
    done
    
    # Alignment Loss Lambda variations
    for LAMBDA_ALIGN in 0.1 0.2 0.3 0.5 1.0; do
        submit_job "align_lambda_${LAMBDA_ALIGN}" \
            "LAMBDA_ALIGNMENT=${LAMBDA_ALIGN}"
    done
    
    # Combined Rank + Alignment at best individual values
    for LAMBDA_RANK in 0.1 0.2 0.3; do
        for LAMBDA_ALIGN in 0.1 0.2 0.3; do
            submit_job "rank${LAMBDA_RANK}_align${LAMBDA_ALIGN}" \
                "LAMBDA_RANK=${LAMBDA_RANK} LAMBDA_ALIGNMENT=${LAMBDA_ALIGN}"
        done
    done
    
    # Text Contrastive Lambda variations
    for LAMBDA_TC in 0.1 0.2 0.3 0.5; do
        submit_job "textcontr_lambda_${LAMBDA_TC}" \
            "LAMBDA_TEXT_CONTRASTIVE=${LAMBDA_TC}"
    done
    
    # Component Loss Lambda variations
    for LAMBDA_COMP in 0.1 0.25 0.5 0.75 1.0; do
        submit_job "comp_lambda_${LAMBDA_COMP}" \
            "LAMBDA_COMPONENTS=${LAMBDA_COMP}"
    done
}

# =============================================================================
# ABLATION 2: Number of Components per Step
# =============================================================================

run_component_count_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 2: Number of Components per Step"
    echo "=============================================="
    
    # NUM_COMPONENT_CAPTIONS: how many component captions per sample
    for N_C in 1 3 4; do
        submit_job "num_comp_captions_${N_C}" \
            "NUM_COMPONENT_CAPTIONS=${N_C}"
    done
}

# =============================================================================
# ABLATION 3: Max Components Parameter (K)
# =============================================================================

run_max_components_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 3: Max Components (K)"
    echo "=============================================="
    
    # MAX_COMPONENTS_PER_SAMPLE: maximum components to construct
    for K in 1 2 3 5; do
        submit_job "max_comp_K${K}" \
            "MAX_COMPONENTS_PER_SAMPLE=${K}"
    done
    
    # Also try combinations with different num_component_captions
    for K in 2 3; do
        for N_C in 1 2 3; do
            submit_job "K${K}_nc${N_C}" \
                "MAX_COMPONENTS_PER_SAMPLE=${K} NUM_COMPONENT_CAPTIONS=${N_C}"
        done
    done
}

# =============================================================================
# ABLATION 4: Learning Rate, Batch Size, Epochs
# =============================================================================

run_optimization_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 4: Learning Rate & Batch Size"
    echo "=============================================="
    
    # Learning rate sweep
    for LR in "1e-6" "2e-6" "5e-6" "1e-5" "2e-5"; do
        submit_job "lr_${LR}" \
            "LR=${LR}"
    done
    
    # Batch size sweep (with adjusted LR using linear scaling)
    for BS in 32 64 128; do
        # Linear scaling: LR_new = LR_base * (BS_new / BS_base)
        LR_SCALED=$(python3 -c "print(f'{5e-6 * ${BS} / 128:.1e}')")
        submit_job "bs${BS}_lr${LR_SCALED}" \
            "BATCH_SIZE=${BS} LR=${LR_SCALED}"
    done
    
    # Weight decay sweep
    for WD in "1e-3" "5e-3" "1e-2" "2e-2" "5e-2"; do
        submit_job "wd_${WD}" \
            "WD=${WD}"
    done
}

# =============================================================================
# ABLATION 5: Paraphrase Loss (ReadCLIP-style)
# =============================================================================

run_paraphrase_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 5: Paraphrase Loss"
    echo "=============================================="
    
    # Paraphrase loss lambda variations
    for LAMBDA_PARA in 0.1 0.2 0.3 0.5; do
        submit_job "paraphrase_${LAMBDA_PARA}" \
            "LAMBDA_PARAPHRASE=${LAMBDA_PARA}"
    done
}

# =============================================================================
# ABLATION 6: Negative Sampling Strategies
# =============================================================================

run_negative_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 6: Negative Sampling Strategies"
    echo "=============================================="
    
    # Swap negatives probability
    for SWAP_P in 0.0 0.25 0.5 0.75; do
        for INPLACE_P in 1.0 0.5 0.0; do
            submit_job "swap${SWAP_P}_inplace${INPLACE_P}" \
                "USE_SWAP_NEGATIVES=true SWAP_NEGATIVE_PROB=${SWAP_P} INPLACE_REPLACEMENT_PROB=${INPLACE_P}"
        done
    done
    
}

# =============================================================================
# ABLATION 7: Sampling Strategies
# =============================================================================

run_sampling_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 7: Structured Sampling Strategies"
    echo "=============================================="
    
    # -------------------------------------------------------------------------
    # 7.1: Structured vs Legacy sampling
    # -------------------------------------------------------------------------
    echo "  7.1: Structured vs Legacy"
    submit_job "structured_true" \
        "USE_STRUCTURED_SAMPLING=true"
    submit_job "structured_false" \
        "USE_STRUCTURED_SAMPLING=false"
    
    # -------------------------------------------------------------------------
    # 7.2: Component-only vs Relation-only vs Binding-only
    # These test each sampling type in isolation
    # -------------------------------------------------------------------------
    echo "  7.2: Isolated sampling types"
    
    # Components only (STRUCTURED_RELATION_PROB=0 means always components, no bindings)
    submit_job "components_only" \
        "USE_STRUCTURED_SAMPLING=true STRUCTURED_RELATION_PROB=0.0 BINDING_NEGATIVE_PROB=0.0"
    
    # Relations only (STRUCTURED_RELATION_PROB=1.0 means always try relations first)
    submit_job "relations_only" \
        "USE_STRUCTURED_SAMPLING=true STRUCTURED_RELATION_PROB=1.0 BINDING_NEGATIVE_PROB=0.0"
    
    # Bindings only (high binding prob, will fall back to others if no binding available)
    submit_job "bindings_only" \
        "USE_STRUCTURED_SAMPLING=true BINDING_NEGATIVE_PROB=1.0"
    
    # -------------------------------------------------------------------------
    # 7.3: Equal probability mixtures
    # -------------------------------------------------------------------------
    echo "  7.3: Equal probability mixtures"
    
    # 50-50 components vs relations (no bindings)
    submit_job "comp_rel_50_50" \
        "USE_STRUCTURED_SAMPLING=true STRUCTURED_RELATION_PROB=0.5 BINDING_NEGATIVE_PROB=0.0"
    
    # Equal chance: 1/3 binding, then 50-50 comp/rel for remaining
    # P(binding)=0.33, P(relation|not binding)=0.5, P(component|not binding)=0.5
    submit_job "equal_all_three" \
        "USE_STRUCTURED_SAMPLING=true BINDING_NEGATIVE_PROB=0.33 STRUCTURED_RELATION_PROB=0.5"
    
    # -------------------------------------------------------------------------
    # 7.4: Binding probability sweep
    # -------------------------------------------------------------------------
    echo "  7.4: Binding probability sweep"
    for BIND_P in 0.1 0.2 0.3 0.5 0.7; do
        submit_job "binding_prob_${BIND_P}" \
            "USE_STRUCTURED_SAMPLING=true BINDING_NEGATIVE_PROB=${BIND_P} STRUCTURED_RELATION_PROB=0.5"
    done
    
    # -------------------------------------------------------------------------
    # 7.5: Relation probability sweep (with fixed binding prob)
    # -------------------------------------------------------------------------
    echo "  7.5: Relation probability sweep"
    for REL_P in 0.0 0.25 0.5 0.75 1.0; do
        submit_job "rel_prob_${REL_P}" \
            "USE_STRUCTURED_SAMPLING=true STRUCTURED_RELATION_PROB=${REL_P} BINDING_NEGATIVE_PROB=0.0"
    done
    
    # -------------------------------------------------------------------------
    # 7.6: Context in component pairs
    # -------------------------------------------------------------------------
    echo "  7.6: Context in component pairs"
    submit_job "context_true" \
        "USE_STRUCTURED_SAMPLING=true USE_CONTEXT_IN_COMPONENT_PAIRS=true"
    submit_job "context_false" \
        "USE_STRUCTURED_SAMPLING=true USE_CONTEXT_IN_COMPONENT_PAIRS=false"
    
    # -------------------------------------------------------------------------
    # 7.7: Combined binding + relation grid search
    # -------------------------------------------------------------------------
    echo "  7.7: Binding + Relation grid"
    for BIND_P in 0.2 0.4; do
        for REL_P in 0.3 0.5 0.7; do
            submit_job "bind${BIND_P}_rel${REL_P}" \
                "USE_STRUCTURED_SAMPLING=true BINDING_NEGATIVE_PROB=${BIND_P} STRUCTURED_RELATION_PROB=${REL_P}"
        done
    done
}

# =============================================================================
# ABLATION 8: Loss Type Variations
# =============================================================================

run_loss_type_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 8: Loss Types"
    echo "=============================================="
    
    # Contrastive mode variations
    for MODE in "with_components" "with_components_negatives" "without_components"; do
        submit_job "contrastive_mode_${MODE}" \
            "CONTRASTIVE_MODE=${MODE}"
    done
    
    # Component loss type variations
    for COMP_LOSS in "clip" "negclip" "negclip_hard"; do
        submit_job "comp_loss_${COMP_LOSS}" \
            "COMPONENT_LOSS_TYPE=${COMP_LOSS}"
    done
    
    # Alignment loss type (with alignment enabled)
    for ALIGN_TYPE in "cosine" "margin"; do
        submit_job "align_type_${ALIGN_TYPE}" \
            "ALIGNMENT_LOSS_TYPE=${ALIGN_TYPE} LAMBDA_ALIGNMENT=0.2"
    done
    
    # Alignment margin variations (only when using margin type)
    for MARGIN in 0.1 0.2 0.3 0.5; do
        submit_job "align_margin_${MARGIN}" \
            "ALIGNMENT_LOSS_TYPE=margin ALIGNMENT_MARGIN=${MARGIN} LAMBDA_ALIGNMENT=0.2"
    done
    
    # Rank margin variations (with rank enabled)
    for RANK_M in 0.05 0.1 0.2 0.3; do
        submit_job "rank_margin_${RANK_M}" \
            "RANK_MARGIN=${RANK_M} LAMBDA_RANK=0.2"
    done
    
    # Text contrastive margin (with text contrastive enabled)
    for TC_M in 0.1 0.2 0.3 0.5; do
        submit_job "tc_margin_${TC_M}" \
            "TEXT_CONTRASTIVE_MARGIN=${TC_M} LAMBDA_TEXT_CONTRASTIVE=0.2"
    done
}

# =============================================================================
# ABLATION 9: Combined Best Settings Analysis
# =============================================================================
# Based on findings:
# - Binding=0.3, Relation=0.7 works well
# - Swap=0.5, Inplace=1.0 helps
# - Paraphrase=0.1 helps
# - Rank and Align losses can also help

run_combined_best_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 9: Combined Best Settings"
    echo "=============================================="
    
    # Best sampling settings (base for all experiments)
    BEST_SAMPLING="BINDING_NEGATIVE_PROB=0.3 STRUCTURED_RELATION_PROB=0.7 SWAP_NEGATIVE_PROB=0.5 INPLACE_REPLACEMENT_PROB=1.0 USE_STRUCTURED_SAMPLING=true"
    
    # -------------------------------------------------------------------------
    # 9.1: Baseline with best sampling (no auxiliary losses)
    # -------------------------------------------------------------------------
    echo "  9.1: Best sampling baseline"
    submit_job "best_baseline" \
        "${BEST_SAMPLING}"
    
    # -------------------------------------------------------------------------
    # 9.2: + Single auxiliary loss
    # -------------------------------------------------------------------------
    echo "  9.2: Single auxiliary loss additions"
    
    # + Paraphrase only
    submit_job "best_para0.1" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1"
    submit_job "best_para0.2" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.2"
    
    # + Rank only
    submit_job "best_rank0.1" \
        "${BEST_SAMPLING} LAMBDA_RANK=0.1"
    submit_job "best_rank0.2" \
        "${BEST_SAMPLING} LAMBDA_RANK=0.2"
    
    # + Alignment only
    submit_job "best_align0.1" \
        "${BEST_SAMPLING} LAMBDA_ALIGNMENT=0.1"
    submit_job "best_align0.2" \
        "${BEST_SAMPLING} LAMBDA_ALIGNMENT=0.2"
    
    # -------------------------------------------------------------------------
    # 9.3: + Two auxiliary losses (pairwise combinations)
    # -------------------------------------------------------------------------
    echo "  9.3: Pairwise auxiliary loss combinations"
    
    # Paraphrase + Rank
    submit_job "best_para0.1_rank0.1" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_RANK=0.1"
    submit_job "best_para0.1_rank0.2" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_RANK=0.2"
    
    # Paraphrase + Alignment
    submit_job "best_para0.1_align0.1" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_ALIGNMENT=0.1"
    submit_job "best_para0.1_align0.2" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_ALIGNMENT=0.2"
    
    # Rank + Alignment
    submit_job "best_rank0.1_align0.1" \
        "${BEST_SAMPLING} LAMBDA_RANK=0.1 LAMBDA_ALIGNMENT=0.1"
    submit_job "best_rank0.1_align0.2" \
        "${BEST_SAMPLING} LAMBDA_RANK=0.1 LAMBDA_ALIGNMENT=0.2"
    
    # -------------------------------------------------------------------------
    # 9.4: + All three auxiliary losses
    # -------------------------------------------------------------------------
    echo "  9.4: All auxiliary losses combined"
    
    submit_job "best_all_0.1" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_RANK=0.1 LAMBDA_ALIGNMENT=0.1"
    submit_job "best_all_mixed" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_RANK=0.2 LAMBDA_ALIGNMENT=0.1"
    
    # -------------------------------------------------------------------------
    # 9.5: + Text Contrastive (exploratory)
    # -------------------------------------------------------------------------
    echo "  9.5: Text contrastive exploration"
    
    submit_job "best_tc0.1" \
        "${BEST_SAMPLING} LAMBDA_TEXT_CONTRASTIVE=0.1"
    submit_job "best_all_tc" \
        "${BEST_SAMPLING} LAMBDA_PARAPHRASE=0.1 LAMBDA_RANK=0.1 LAMBDA_ALIGNMENT=0.1 LAMBDA_TEXT_CONTRASTIVE=0.1"
}

run_clip_component_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 10: CLIP + Component Loss (No NegCLIP Full)"
    echo "=============================================="
    
    # Base config: CLIP loss on full, negclip_hard on components
    CLIP_COMP_BASE="USE_NEGATIVES_FULL=false CONTRASTIVE_MODE=with_components_negatives COMPONENT_LOSS_TYPE=negclip_hard"
    
    # Baseline: CLIP + Component (negclip_hard)
    # submit_job "clip_comp_baseline" \
    #     "${CLIP_COMP_BASE}"
    
    # # Compare component loss types
    # submit_job "clip_comp_clip" \
    #     "USE_NEGATIVES_FULL=false CONTRASTIVE_MODE=with_components_negatives COMPONENT_LOSS_TYPE=clip"
    # submit_job "clip_comp_negclip" \
    #     "USE_NEGATIVES_FULL=false CONTRASTIVE_MODE=with_components_negatives COMPONENT_LOSS_TYPE=negclip"

    # Use swap negatives but only components
    submit_job "negclip_w_comp" \
        "USE_NEGATIVES_FULL=true CONTRASTIVE_MODE=with_components_negatives COMPONENT_LOSS_TYPE=clip"

    submit_job "clip_ft" \
        "USE_NEGATIVES_FULL=false CONTRASTIVE_MODE=without_negatives COMPONENT_LOSS_TYPE=clip"
}




# =============================================================================
# ABLATION 11: Encoder Freezing (Text-only vs Image-only fine-tuning)
# =============================================================================
# Tests whether fine-tuning only one encoder is sufficient:
# - Freeze text, fine-tune image only
# - Freeze image, fine-tune text only
# - Baseline: fine-tune both (default)

run_encoder_freeze_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 11: Encoder Freezing"
    echo "=============================================="
    
    # These ablations require modifying the training script call
    # We need a custom submit function that passes --ft-text/--ft-image flags
    
    # -------------------------------------------------------------------------
    # 11.1: Freeze Text Encoder (Fine-tune Image Only)
    # -------------------------------------------------------------------------
    echo "  11.1: Freeze text encoder (fine-tune image only)"
    submit_job_custom "freeze_text_ft_image" \
        "" \
        "--ft-text false --ft-image true"
    
    # -------------------------------------------------------------------------
    # 11.2: Freeze Image Encoder (Fine-tune Text Only)
    # -------------------------------------------------------------------------
    echo "  11.2: Freeze image encoder (fine-tune text only)"
    submit_job_custom "freeze_image_ft_text" \
        "" \
        "--ft-text true --ft-image false"

}


# =============================================================================
# ABLATION 12: Model Sizes (ViT-B/32 vs ViT-L/14)
# =============================================================================
# Tests different CLIP backbone sizes:
# - ViT-B/32 (512-dim, faster, baseline)
# - ViT-B/16 (512-dim, higher resolution)
# - ViT-L/14 (768-dim, larger capacity, needs smaller batch size)

run_model_size_ablations() {
    echo ""
    echo "=============================================="
    echo "ABLATION 12: Model Sizes"
    echo "=============================================="
    
    # -------------------------------------------------------------------------
    # 12.1: ViT-B/32 with OpenAI weights (baseline)
    # -------------------------------------------------------------------------
    echo "  12.1: ViT-B/32 (OpenAI weights) - baseline"
    submit_job_custom "model_vit_b32_openai" \
        "MODEL=vit_b32_openai BATCH_SIZE=128" \
        "--ft-both"
    
    # -------------------------------------------------------------------------
    # 12.2: ViT-B/16 with OpenAI weights (higher resolution)
    # -------------------------------------------------------------------------
    echo "  12.2: ViT-B/16 (OpenAI weights) - higher resolution"
    submit_job_custom "model_vit_b16_openai" \
        "MODEL=vit_b16_openai BATCH_SIZE=96" \
        "--ft-both"
    
    # -------------------------------------------------------------------------
    # 12.3: ViT-L/14 with OpenAI weights (larger model, smaller batch)
    # -------------------------------------------------------------------------
    echo "  12.3: ViT-L/14 (OpenAI weights) - larger model"
    submit_job_custom "model_vit_l14_openai" \
        "MODEL=vit_l14_openai BATCH_SIZE=64" \
        "--ft-both"
}


# =============================================================================
# CUSTOM SUBMIT FUNCTION (for encoder freezing and model size ablations)
# =============================================================================
# This function allows passing custom flags to train_laion_ft.sh
# instead of just environment variables to train_structured.sh

submit_job_custom() {
    local JOB_NAME="$1"
    local EXTRA_ENV_VARS="$2"    # Environment variables (VAR=value pairs)
    local SCRIPT_FLAGS="$3"      # Flags for train_laion_ft.sh (e.g., --ft-text true)
    
    JOB_COUNT=$((JOB_COUNT + 1))
    
    # Create SLURM script
    local SLURM_SCRIPT=$(mktemp /tmp/ablation_${JOB_NAME}_XXXXX.slurm)
    
    # Determine batch size and model from env vars (for memory estimation)
    local BATCH_SIZE_VAL=$(echo "$EXTRA_ENV_VARS" | grep -oP 'BATCH_SIZE=\K[0-9]+' || echo "128")
    local MODEL_VAL=$(echo "$EXTRA_ENV_VARS" | grep -oP 'MODEL=\K[a-z0-9_]+' || echo "vit_b32_openai")
    
    # Adjust memory for larger models
    local MEM_MULTIPLIER=32
    if [[ "$MODEL_VAL" == *"l14"* ]]; then
        MEM_MULTIPLIER=48  # ViT-L/14 needs more memory
    fi
    
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$((GPUS_PER_JOB * CPUS_PER_GPU))
#SBATCH --gres=gpu:${GPUS_PER_JOB}
#SBATCH --mem=$((GPUS_PER_JOB * MEM_MULTIPLIER))G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/ablation_%x_%j.out
#SBATCH --error=logs/ablation_%x_%j.err
#SBATCH --mail-type=END,FAIL

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "Job: \${SLURM_JOB_NAME}"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node: \${SLURM_NODELIST}"
echo "GPUs: ${GPUS_PER_JOB}"
echo "Model: ${MODEL_VAL}"
echo "Batch Size: ${BATCH_SIZE_VAL}"
echo "Start Time: \$(date)"
echo "=========================================="

# Change to working directory
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate /mnt/lustre/work/oh/owl336/.conda/py-311-pytorch

# Set environment variables
export ${EXTRA_ENV_VARS}

# Run training with custom script flags
# Note: We bypass train_structured.sh and call train_laion_ft.sh directly
# for more control over encoder fine-tuning flags

./train_laion_ft.sh \\
    -y \\
    --skip-tar-range \\
    ${SCRIPT_FLAGS} \\
    --model \${MODEL:-vit_b32_openai} \\
    --gpus ${GPUS_PER_JOB} \\
    --batch-size \${BATCH_SIZE:-128} \\
    --amp \\
    --optimizer adamw_ft \\
    --lr \${LR:-5e-6} \\
    --wd \${WD:-1e-2} \\
    --name "ablation_${JOB_NAME}" \\
    --eval-csv "\$(date +%d-%b)_ablation_${JOB_NAME}.csv" \\
    training.force_float32=true \\
    loss=multi_caption \\
    loss.contrastive_mode="\${CONTRASTIVE_MODE:-with_components_negatives}" \\
    loss.lambda_full=\${LAMBDA_FULL:-1.0} \\
    loss.use_negatives_full=\${USE_NEGATIVES_FULL:-true} \\
    loss.lambda_components=\${LAMBDA_COMPONENTS:-0.5} \\
    loss.component_loss_type="\${COMPONENT_LOSS_TYPE:-negclip_hard}" \\
    dataset=coco_neg \\
    dataset.dataset_kwargs.json_folder="/mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally/binding_pos_json/coco_train_v2/" \\
    dataset.dataset_kwargs.image_root="." \\
    dataset.dataset_kwargs.use_structured_sampling=true \\
    dataset.dataset_kwargs.structured_relation_prob=1.0 \\
    cache.cache_folder=coco_cache \\
    head=bimodal \\
    evaluation.initial_evaluate=false \\
    training.use_amp=true \\
    dataset.val_ratio=0.001 \\
    dataset.subset_name='train'

echo "=========================================="
echo "End Time: \$(date)"
echo "=========================================="
EOF

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "=========================================="
        echo "[DRY RUN] Would submit job #${JOB_COUNT}: ${JOB_NAME}"
        echo "=========================================="
        echo "Env vars: ${EXTRA_ENV_VARS}"
        echo "Script flags: ${SCRIPT_FLAGS}"
        echo "Script: ${SLURM_SCRIPT}"
        cat "$SLURM_SCRIPT"
    else
        echo ""
        echo "Submitting job #${JOB_COUNT}: ${JOB_NAME}"
        JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
        SUBMITTED_JOBS+=("${JOB_ID}:${JOB_NAME}")
        echo "  -> Job ID: ${JOB_ID}"
    fi
    
    rm -f "$SLURM_SCRIPT"
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=============================================="
echo "ABLATION STUDY LAUNCHER"
echo "=============================================="
echo "Partition:    ${PARTITION}"
echo "Time Limit:   ${TIME_LIMIT}"
echo "GPUs/Job:     ${GPUS_PER_JOB}"
echo "Ablation:     ${ABLATION_FILTER}"
echo "Dry Run:      ${DRY_RUN}"
echo "=============================================="

# Create logs directory
mkdir -p logs

# Run selected ablations
case $ABLATION_FILTER in
    "lambdas")
        run_lambda_ablations
        ;;
    "components")
        run_component_count_ablations
        # run_max_components_ablations
        ;;
    "lr"|"optimization")
        run_optimization_ablations
        ;;
    "negatives")
        run_negative_ablations
        ;;
    "paraphrase")
        run_paraphrase_ablations
        ;;
    "sampling")
        run_sampling_ablations
        ;;
    "loss_types")
        run_loss_type_ablations
        ;;
    "combined")
        run_combined_best_ablations
        ;;
    "clip_component")
        run_clip_component_ablations
        ;;
    "encoder_freeze"|"freeze")
        run_encoder_freeze_ablations
        ;;
    "model_size"|"models")
        run_model_size_ablations
        ;;
    "all")
        run_lambda_ablations
        run_component_count_ablations
        run_max_components_ablations
        run_optimization_ablations
        run_paraphrase_ablations
        run_negative_ablations
        run_sampling_ablations
        run_loss_type_ablations
        run_combined_best_ablations
        run_encoder_freeze_ablations
        run_model_size_ablations
        ;;
    *)
        echo "Unknown ablation: ${ABLATION_FILTER}"
        echo "Available: lambdas, components, lr, negatives, paraphrase, sampling, loss_types, combined, encoder_freeze, model_size, all"
        exit 1
        ;;
esac

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Total jobs: ${JOB_COUNT}"

if [ "$DRY_RUN" = false ] && [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "Submitted Jobs:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo "  ${job}"
    done
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all:   scancel -u \$USER -n 'ablation_*'"
fi

echo "=============================================="
