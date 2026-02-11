#!/bin/bash
# filepath: /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally/train_structured.sh
# Unified training script for structured negative training with LAION or COCO datasets
set -e

###########################################
# COMMAND LINE ARGUMENT PARSING
###########################################

# Default values
NO_EVAL=false
DATASET="laion"  # "laion" or "coco"
GPUS=1
HEAD="bimodal"   # "bimodal" or "text_query_aggregator"
TAG=""           # Optional tag to append to experiment name

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-eval)
            NO_EVAL=true
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset <laion|coco>    Dataset to use (default: laion)"
            echo "  --gpus <N>                Number of GPUs (default: 1)"
            echo "  --head <bimodal|text_query_aggregator>  Head type (default: bimodal)"
            echo "  --tag <string>            Optional tag to append to experiment name"
            echo "  --no-eval                 Disable evaluation (test mode)"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate dataset choice
if [[ "$DATASET" != "laion" && "$DATASET" != "coco" ]]; then
    echo "Error: Invalid dataset '$DATASET'. Choose 'laion' or 'coco'"
    exit 1
fi

###########################################
# USER-ADJUSTABLE EXPERIMENT PARAMETERS
# NOTE: All parameters can be overridden via environment variables
# Example: LAMBDA_RANK=0.5 ./train_structured.sh --dataset coco
###########################################

MODEL="${MODEL:-vit_b32_openai}"

# LAION-specific settings (only used when DATASET=laion)
TAR_RANGE="${TAR_RANGE:-[0,16]}"

# Common sampling settings
NUM_COMPONENT_CAPTIONS="${NUM_COMPONENT_CAPTIONS:-2}"
MAX_COMPONENTS_PER_SAMPLE="${MAX_COMPONENTS_PER_SAMPLE:-2}"
MAX_POSITIVE_COMPONENTS_WITH_NEG="${MAX_POSITIVE_COMPONENTS_WITH_NEG:-1}"

# Sampling mode
USE_STRUCTURED_SAMPLING="${USE_STRUCTURED_SAMPLING:-true}"       # Enable structured paired positive-negative sampling
STRUCTURED_RELATION_PROB="${STRUCTURED_RELATION_PROB:-1.0}"       # Probability of trying relation pairs first (vs component pairs)
USE_CONTEXT_IN_COMPONENT_PAIRS="${USE_CONTEXT_IN_COMPONENT_PAIRS:-true}" # Include other components for context in component pairs
BINDING_NEGATIVE_PROB="${BINDING_NEGATIVE_PROB:-0.0}"             # Probability of sampling binding negatives (noun swaps between components)

# Loss configuration
LAMBDA_FULL="${LAMBDA_FULL:-1.0}"                        # Weight for full caption loss
USE_NEGATIVES_FULL="${USE_NEGATIVES_FULL:-true}"                # Use NegCLIP (true) or CLIP (false) for full caption
LAMBDA_COMPONENTS="${LAMBDA_COMPONENTS:-0.5}"                  # Weight for component contrastive loss
LAMBDA_ALIGNMENT="${LAMBDA_ALIGNMENT:-0.0}"                   # Weight for alignment loss (separate from contrastive)
LAMBDA_PARAPHRASE="${LAMBDA_PARAPHRASE:-0.0}"                  # Weight for paraphrase loss
LAMBDA_RANK="${LAMBDA_RANK:-0.0}"                        # Weight for ranking loss (full > components)
RANK_MARGIN="${RANK_MARGIN:-0.1}"                        # Margin for ranking: s(full) > s(comp) + margin
LAMBDA_TEXT_CONTRASTIVE="${LAMBDA_TEXT_CONTRASTIVE:-0.0}"            # Weight for text contrastive margin loss
TEXT_CONTRASTIVE_MARGIN="${TEXT_CONTRASTIVE_MARGIN:-0.2}"            # Margin for text contrastive: s(t_full, t_comp+) > s(t_full, t_comp-) + margin

COMPONENT_LOSS_TYPE="${COMPONENT_LOSS_TYPE:-negclip_hard}"          # "clip", "negclip", or "negclip_hard"
                                       # - clip: Standard CLIP loss
                                       # - negclip: batch hard negatives (B+B scores in denominator)
                                       # - negclip_hard: per-sample hard negative only (B+1 scores)
ALIGNMENT_LOSS_TYPE="${ALIGNMENT_LOSS_TYPE:-margin}"           # "cosine" or "margin"
                                       # - cosine: L = 1 - cos(I, t_k)
                                       # - margin: L = max(0, m - s_pos + s_neg)
ALIGNMENT_MARGIN="${ALIGNMENT_MARGIN:-0.2}"                   # Margin for alignment_margin loss
CONTRASTIVE_MODE="${CONTRASTIVE_MODE:-with_components_negatives}"  # Loss mode

# Relation / component sampling
SAMPLE_RELATIONS="${SAMPLE_RELATIONS:-true}"
SAMPLE_REL_OR_COMP="${SAMPLE_REL_OR_COMP:-mixed}"        # "mixed", "relation_only", "components_only"
MIX_COMP_REL_PROB="${MIX_COMP_REL_PROB:-0.0}"             # 0.0 = disable comp+rel mixing (acts like "either")
RELATION_SAMPLE_PROB="${RELATION_SAMPLE_PROB:-0.5}"

# Negative sampling settings
NEG_REL_PROB="${NEG_REL_PROB:-0.2}"
INPLACE_REPLACEMENT_PROB="${INPLACE_REPLACEMENT_PROB:-1.0}"
USE_SWAP_NEGATIVES="${USE_SWAP_NEGATIVES:-true}"
SWAP_NEGATIVE_PROB="${SWAP_NEGATIVE_PROB:-1.0}"

# Optimization
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-6}"
WD="${WD:-1e-2}"

# COCO-specific settings (only used when DATASET=coco)
COCO_JSON_FOLDER="/mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally/binding_pos_json/coco_train_v2/"
COCO_IMAGE_ROOT="."

###########################################
# AUTO-GENERATE EXPERIMENT NAME AND CSV
###########################################

# Function to format scientific notation to human-readable
format_lr() {
    python3 -c "print(f'{float(\"$1\"):.0e}'.replace('-0', 'e-').replace('e+0', 'e+'))"
}

format_wd() {
    python3 -c "print(f'{float(\"$1\"):.0e}'.replace('e-0', 'e-').replace('e+0', 'e+'))"
}

# Extract date
DATE=$(date +%d-%b)

# Build sampling strategy name
SAMPLING_MODE="legacy"
if [ "$USE_STRUCTURED_SAMPLING" = true ]; then
    SAMPLING_MODE="structured_rel${STRUCTURED_RELATION_PROB}"
fi

if [ "$SAMPLE_REL_OR_COMP" = "mixed" ]; then
    if (( $(echo "$MIX_COMP_REL_PROB == 0.0" | bc -l) )); then
        SAMPLING_STRATEGY="either"
    else
        SAMPLING_STRATEGY="mixed_p${MIX_COMP_REL_PROB}"
    fi
elif [ "$SAMPLE_REL_OR_COMP" = "relation_only" ]; then
    SAMPLING_STRATEGY="rel_only"
elif [ "$SAMPLE_REL_OR_COMP" = "components_only" ]; then
    SAMPLING_STRATEGY="comp_only"
else
    SAMPLING_STRATEGY="pos_${SAMPLE_REL_OR_COMP}"
fi

# Build negative strategy name
NEG_STRATEGY="neg_rel${NEG_REL_PROB}_inplace${INPLACE_REPLACEMENT_PROB}"

# Add swap negatives to strategy if enabled
if [ "$USE_SWAP_NEGATIVES" = true ]; then
    NEG_STRATEGY="${NEG_STRATEGY}_swap${SWAP_NEGATIVE_PROB}"
fi

# Format learning rate and weight decay
LR_STR=$(format_lr $LR)
WD_STR=$(format_wd $WD)

# Shorten contrastive mode for experiment name
if [ "$CONTRASTIVE_MODE" = "without_components" ]; then
    MODE_STR="no_comp"
elif [ "$CONTRASTIVE_MODE" = "with_components" ]; then
    MODE_STR="w_comp"
else
    MODE_STR="${CONTRASTIVE_MODE}"
fi

# Construct experiment name with dataset prefix
EXP_PREFIX=""
if [ "$NO_EVAL" = true ]; then
    EXP_PREFIX="test_"
fi

DATASET_PREFIX="${DATASET}"

# Build loss string for experiment name
LOSS_STR="lf${LAMBDA_FULL}_lc${LAMBDA_COMPONENTS}"
if (( $(echo "$LAMBDA_ALIGNMENT > 0" | bc -l) )); then
    LOSS_STR="${LOSS_STR}_la${LAMBDA_ALIGNMENT}"
fi
if (( $(echo "$LAMBDA_RANK > 0" | bc -l) )); then
    LOSS_STR="${LOSS_STR}_lr${LAMBDA_RANK}"
fi
if (( $(echo "$LAMBDA_TEXT_CONTRASTIVE > 0" | bc -l) )); then
    LOSS_STR="${LOSS_STR}_ltc${LAMBDA_TEXT_CONTRASTIVE}"
fi

EXP_NAME="${EXP_PREFIX}${DATASET_PREFIX}_${MODE_STR}_${SAMPLING_MODE}_${SAMPLING_STRATEGY}_max${MAX_COMPONENTS_PER_SAMPLE}_${LOSS_STR}_${COMPONENT_LOSS_TYPE}_lr${LR_STR}_wd${WD_STR}_${NEG_STRATEGY}"

# Append tag if provided
if [ -n "$TAG" ]; then
    EXP_NAME="${EXP_NAME}_${TAG}"
fi

# Construct CSV filename
EVAL_CSV="${DATE}_${EXP_NAME}.csv"

###########################################
# PRINT EXPERIMENT CONFIGURATION
###########################################

echo "=========================================="
echo "EXPERIMENT CONFIGURATION"
echo "=========================================="
echo "Experiment Name: ${EXP_NAME}"
echo "CSV Output:      ${EVAL_CSV}"
echo ""
echo "Dataset:         ${DATASET}"
if [ "$DATASET" = "coco" ]; then
    echo "  JSON Folder:   ${COCO_JSON_FOLDER}"
    echo "  Image Root:    ${COCO_IMAGE_ROOT}"
else
    echo "  TAR Range:     ${TAR_RANGE}"
fi
echo ""
echo "Model:           ${MODEL}"
echo "Head:            ${HEAD}"
echo "Learning Rate:   ${LR} (${LR_STR})"
echo "Weight Decay:    ${WD} (${WD_STR})"
echo "Batch Size:      ${BATCH_SIZE}"
echo "GPUs:            ${GPUS}"
echo ""
echo "Loss Configuration:"
echo "  Contrastive Mode:  ${CONTRASTIVE_MODE}"
echo "  --- Full Caption Loss ---"
echo "  Lambda Full:       ${LAMBDA_FULL}"
echo "  Use Negatives:     ${USE_NEGATIVES_FULL}"
echo "  --- Component Contrastive Loss ---"
echo "  Lambda Components: ${LAMBDA_COMPONENTS}"
echo "  Component Type:    ${COMPONENT_LOSS_TYPE}"
echo "  --- Alignment Loss ---"
echo "  Lambda Alignment:  ${LAMBDA_ALIGNMENT}"
if [ "$(echo "${LAMBDA_ALIGNMENT} > 0" | bc)" = "1" ]; then
    echo "  Alignment Type:    ${ALIGNMENT_LOSS_TYPE}"
    echo "  Alignment Margin:  ${ALIGNMENT_MARGIN}"
fi
echo "  --- Ranking Loss ---"
echo "  Lambda Rank:       ${LAMBDA_RANK}"
if [ "$(echo "${LAMBDA_RANK} > 0" | bc)" = "1" ]; then
    echo "  Rank Margin:       ${RANK_MARGIN}"
fi
echo "  --- Text Contrastive Loss ---"
echo "  Lambda Text Contr: ${LAMBDA_TEXT_CONTRASTIVE}"
if [ "$(echo "${LAMBDA_TEXT_CONTRASTIVE} > 0" | bc)" = "1" ]; then
    echo "  Text Contr Margin: ${TEXT_CONTRASTIVE_MARGIN}"
fi
echo "  --- Paraphrase Loss ---"
echo "  Lambda Paraphrase: ${LAMBDA_PARAPHRASE}"
echo ""
echo "Sampling Mode:       ${SAMPLING_MODE}"
echo "  Use Structured:    ${USE_STRUCTURED_SAMPLING}"
if [ "$USE_STRUCTURED_SAMPLING" = true ]; then
    echo "  Relation Prob:     ${STRUCTURED_RELATION_PROB}"
    echo "  Use Context:       ${USE_CONTEXT_IN_COMPONENT_PAIRS}"
    echo "  Binding Neg Prob:  ${BINDING_NEGATIVE_PROB}"
fi
echo ""
echo "Sampling Strategy: ${SAMPLE_REL_OR_COMP}"
echo "  Mix Prob:        ${MIX_COMP_REL_PROB}"
echo "  Relation Prob:   ${RELATION_SAMPLE_PROB}"
echo "  Max Components:  ${MAX_COMPONENTS_PER_SAMPLE}"
echo ""
echo "Negative Sampling:"
echo "  Neg Rel Prob:    ${NEG_REL_PROB}"
echo "  Inplace Prob:    ${INPLACE_REPLACEMENT_PROB}"
echo "  Use Swap Negs:   ${USE_SWAP_NEGATIVES}"
echo "  Swap Prob:       ${SWAP_NEGATIVE_PROB}"
echo ""
echo "Evaluation:        $(if [ "$NO_EVAL" = true ]; then echo "DISABLED (test mode)"; else echo "ENABLED"; fi)"
echo "=========================================="
echo ""

###########################################
# BUILD DATASET-SPECIFIC ARGUMENTS
###########################################

if [ "$DATASET" = "coco" ]; then
    DATASET_ARGS=(
        "dataset=coco_neg"
        "dataset.dataset_kwargs.json_folder=${COCO_JSON_FOLDER}"
        "dataset.dataset_kwargs.image_root=${COCO_IMAGE_ROOT}"
        "dataset.dataset_kwargs.num_component_captions=${NUM_COMPONENT_CAPTIONS}"
        "dataset.dataset_kwargs.use_structured_sampling=${USE_STRUCTURED_SAMPLING}"
        "dataset.dataset_kwargs.structured_relation_prob=${STRUCTURED_RELATION_PROB}"
        "dataset.dataset_kwargs.use_context_in_component_pairs=${USE_CONTEXT_IN_COMPONENT_PAIRS}"
        "dataset.dataset_kwargs.binding_negative_prob=${BINDING_NEGATIVE_PROB}"
        "dataset.dataset_kwargs.swap_negative_prob=${SWAP_NEGATIVE_PROB}"
        "dataset.dataset_kwargs.inplace_replacement_prob=${INPLACE_REPLACEMENT_PROB}"
        "cache.cache_folder=coco_cache"
    )
else
    # LAION dataset - has more sampling parameters
    DATASET_ARGS=(
        "dataset=laion400m"
        "dataset.dataset_kwargs.tar_range=${TAR_RANGE}"
        "dataset.dataset_kwargs.num_component_captions=${NUM_COMPONENT_CAPTIONS}"
        "dataset.dataset_kwargs.max_components_per_sample=${MAX_COMPONENTS_PER_SAMPLE}"
        "dataset.dataset_kwargs.max_positive_components_with_negative=${MAX_POSITIVE_COMPONENTS_WITH_NEG}"
        "dataset.dataset_kwargs.use_structured_sampling=${USE_STRUCTURED_SAMPLING}"
        "dataset.dataset_kwargs.structured_relation_prob=${STRUCTURED_RELATION_PROB}"
        "dataset.dataset_kwargs.use_context_in_component_pairs=${USE_CONTEXT_IN_COMPONENT_PAIRS}"
        "dataset.dataset_kwargs.binding_negative_prob=${BINDING_NEGATIVE_PROB}"
        "dataset.dataset_kwargs.sample_relations=${SAMPLE_RELATIONS}"
        "dataset.dataset_kwargs.sample_relation_or_components=${SAMPLE_REL_OR_COMP}"
        "dataset.dataset_kwargs.mix_comp_rel_prob=${MIX_COMP_REL_PROB}"
        "dataset.dataset_kwargs.relation_sample_prob=${RELATION_SAMPLE_PROB}"
        "dataset.dataset_kwargs.negative_relation_sample_prob=${NEG_REL_PROB}"
        "dataset.dataset_kwargs.inplace_replacement_prob=${INPLACE_REPLACEMENT_PROB}"
        "dataset.dataset_kwargs.use_swap_negatives=${USE_SWAP_NEGATIVES}"
        "dataset.dataset_kwargs.swap_negative_prob=${SWAP_NEGATIVE_PROB}"
        "cache.cache_folder=laion_cache"
    )
fi

###########################################
# LAUNCH TRAINING
###########################################

# Build optional arguments
OPTIONAL_ARGS=""
if [ "$NO_EVAL" = true ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --no-eval"
fi

# Skip tar_range for non-LAION datasets
if [ "$DATASET" != "laion" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --skip-tar-range"
fi

./train_laion_ft.sh \
    -y \
    --ft-both \
    --model ${MODEL} \
    --gpus ${GPUS} \
    training.force_float32=true \
    --batch-size ${BATCH_SIZE} \
    --amp \
    --optimizer adamw_ft \
    --lr ${LR} \
    --wd ${WD} \
    --name "${EXP_NAME}" \
    --eval-csv "${EVAL_CSV}" \
    loss=multi_caption \
    loss.contrastive_mode="${CONTRASTIVE_MODE}" \
    loss.lambda_full=${LAMBDA_FULL} \
    loss.use_negatives_full=${USE_NEGATIVES_FULL} \
    loss.lambda_components=${LAMBDA_COMPONENTS} \
    loss.component_loss_type="${COMPONENT_LOSS_TYPE}" \
    loss.lambda_alignment=${LAMBDA_ALIGNMENT} \
    loss.alignment_loss_type="${ALIGNMENT_LOSS_TYPE}" \
    loss.alignment_margin=${ALIGNMENT_MARGIN} \
    loss.lambda_rank=${LAMBDA_RANK} \
    loss.rank_margin=${RANK_MARGIN} \
    loss.lambda_text_contrastive=${LAMBDA_TEXT_CONTRASTIVE} \
    loss.text_contrastive_margin=${TEXT_CONTRASTIVE_MARGIN} \
    loss.lambda_paraphrase=${LAMBDA_PARAPHRASE} \
    "${DATASET_ARGS[@]}" \
    ${OPTIONAL_ARGS} \
    head=${HEAD} evaluation.initial_evaluate=false training.use_amp=true dataset.val_ratio=0.001 dataset.subset_name='train'
