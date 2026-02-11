#!/bin/bash

# ============================================================================
# SLURM Job Submission Script for LAION Fine-tuning
# ============================================================================
# This script generates and submits a Slurm batch job that runs train_laion_ft.sh
# Edit the variables below to customize your training run.
#
# Usage:
#   ./scripts/submit_train_laion_ft.sh
#
# Or override specific variables:
#   BATCH_SIZE=256 GPUS=4 ./scripts/submit_train_laion_ft.sh
# ============================================================================

# ------------------------------
# Slurm Configuration
# ------------------------------
PARTITION="${PARTITION:-a100-fat-galvani}"
GPUS="${GPUS:-1}"
TIME="${TIME:-48:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
JOB_NAME="${JOB_NAME:-train_laion_ft}"

# ------------------------------
# Training Configuration (Easy to Edit!)
# ------------------------------
MODEL="${MODEL:-vit_b32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-6}"
WD="${WD:-1e-2}"
OPTIMIZER="${OPTIMIZER:-adamw_ft}"

# Dataset configuration
TAR_RANGE="${TAR_RANGE:-[0,13000]}"
NUM_COMPONENT_CAPTIONS="${NUM_COMPONENT_CAPTIONS:-3}"
SAMPLE_RELATIONS="${SAMPLE_RELATIONS:-true}"

# Loss configuration
LOSS_TYPE="${LOSS_TYPE:-multi_caption}"
CONTRASTIVE_MODE="${CONTRASTIVE_MODE:-with_components_negatives}"
LAMBDA_RANK="${LAMBDA_RANK:-0.0}"
ALPHA="${ALPHA:-0.5}"  # Component weighting (0.5 = equal weight)

# Swap negatives configuration
USE_SWAP_NEGATIVES="${USE_SWAP_NEGATIVES:-false}"
SWAP_NEGATIVE_PROB="${SWAP_NEGATIVE_PROB:-0.3}"

# Experiment naming (with variable expansion)
RUN_NAME="${RUN_NAME:-${CONTRASTIVE_MODE}_tar-13k_lr-${LR}_wd-${WD}}"
# Add time stamp to eval csv

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_CSV="${EVAL_CSV:-experiments_csv/${RUN_NAME}_${TIMESTAMP}.csv}"

# Create experiments_csv directory if it doesn't exist
mkdir -p "experiments_csv"

# Evaluation configuration
EVAL_DATASETS="${EVAL_DATASETS:-[\"SugarCrepe\"]}"
NO_EVAL="${NO_EVAL:-true}"  # Set to false to enable evaluation

# Other flags
FORCE_FLOAT32="${FORCE_FLOAT32:-true}"
USE_AMP="${USE_AMP:-true}"
SAMPLE_LOGGING_NUM="${SAMPLE_LOGGING_NUM:-100}"

# ------------------------------
# Paths
# ------------------------------
WORK_DIR="${WORK_DIR:-/mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./train_laion_ft.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-./slurm_logs}"
CACHE_FOLDER="${CACHE_FOLDER:-laion_cache}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Generate unique job file name
JOB_FILE="${OUTPUT_DIR}/${JOB_NAME}_${TIMESTAMP}.job"

# ------------------------------
# Build Training Command
# ------------------------------
TRAIN_CMD="${TRAIN_SCRIPT} \
--ft-both \
--model ${MODEL} \
cache.cache_folder='${CACHE_FOLDER}' \
--gpus ${GPUS} \
training.force_float32=${FORCE_FLOAT32} \
'dataset.dataset_kwargs.tar_range=${TAR_RANGE}' \
--batch-size ${BATCH_SIZE} \
--optimizer ${OPTIMIZER} \
--lr ${LR} \
--wd ${WD} \
--name \"${RUN_NAME}\" \
--eval-csv \"${EVAL_CSV}\" \
dataset.dataset_kwargs.sample_relations=${SAMPLE_RELATIONS} \
dataset.dataset_kwargs.num_component_captions=${NUM_COMPONENT_CAPTIONS} \
loss=${LOSS_TYPE} \
loss.contrastive_mode='${CONTRASTIVE_MODE}' \
loss.alpha=${ALPHA} \
training.sample_logging_num_samples=${SAMPLE_LOGGING_NUM} \
loss.lambda_rank=${LAMBDA_RANK} \
+dataset.dataset_kwargs.use_swap_negatives=${USE_SWAP_NEGATIVES} \
+dataset.dataset_kwargs.swap_negative_prob=${SWAP_NEGATIVE_PROB}"

# Add --amp flag if enabled
if [ "${USE_AMP}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --amp"
fi

# Add --no-eval flag if disabled
if [ "${NO_EVAL}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --no-eval"
fi

# ------------------------------
# Generate SBATCH Job File
# ------------------------------
cat > "${JOB_FILE}" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${OUTPUT_DIR}/${JOB_NAME}-%j.out
#SBATCH --error=${OUTPUT_DIR}/${JOB_NAME}-%j.err

# Print job info
echo "=========================================="
echo "Job ID: \${SLURM_JOB_ID}"
echo "Job Name: ${JOB_NAME}"
echo "Node: \${SLURM_NODELIST}"
echo "Partition: ${PARTITION}"
echo "GPUs: ${GPUS}"
echo "Start Time: \$(date)"
echo "=========================================="
echo ""

# Change to working directory
cd ${WORK_DIR}

# Optional: Load modules or activate conda environment
# Uncomment and modify as needed:
# module load cuda/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Print environment info
echo "Python: \$(which python)"
echo "GPU Info:"
nvidia-smi
echo ""
echo "=========================================="
echo "Training Command:"
echo "${TRAIN_CMD}"
echo "=========================================="
echo ""

# Run training with srun
srun ${TRAIN_CMD}

# Print completion info
EXIT_CODE=\$?
echo ""
echo "=========================================="
echo "Job finished at: \$(date)"
echo "Exit code: \${EXIT_CODE}"
echo "=========================================="

exit \${EXIT_CODE}
EOF

# ------------------------------
# Submit the job
# ------------------------------
echo "==========================================" echo "Generated job file: ${JOB_FILE}"
echo "==========================================" echo ""
echo "Job Configuration:"
echo "  Partition: ${PARTITION}"
echo "  GPUs: ${GPUS}"
echo "  Time: ${TIME}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Model: ${MODEL}"
echo "  Learning Rate: ${LR}"
echo "  Components: ${NUM_COMPONENT_CAPTIONS}"
echo "  Loss: ${LOSS_TYPE} (${CONTRASTIVE_MODE})"
echo "  Alpha: ${ALPHA}"
echo "  Swap Negatives: ${USE_SWAP_NEGATIVES} (prob=${SWAP_NEGATIVE_PROB})"
echo "  Run Name: ${RUN_NAME}"
echo "==========================================" echo ""

# Submit the job
sbatch "${JOB_FILE}"

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${OUTPUT_DIR}/${JOB_NAME}-<jobid>.out"
else
    echo "✗ Job submission failed!"
    exit 1
fi