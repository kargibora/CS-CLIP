#!/bin/bash
# =============================================================================
# Batch Half-Truth Vulnerability Experiment Runner
# =============================================================================
# This script reads checkpoints from eval_checkpoints.yaml and runs the
# half-truth vulnerability experiment for each one.
#
# Usage:
#   ./run_half_truth_batch.sh                          # Run all checkpoints
#   ./run_half_truth_batch.sh --dry-run                # Show commands without running
#   ./run_half_truth_batch.sh --config other.yaml      # Use different config
#   ./run_half_truth_batch.sh --filter "CLOVE"         # Run only checkpoints matching filter
#   ./run_half_truth_batch.sh --num_gpus 8             # Run in parallel on 8 GPUs
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------

CONFIG_FILE="configs/eval_checkpoints.yaml"
DATASET="laion"
COCO_JSON="swap_pos_json/coco_val/"
COCO_IMAGE_ROOT="."
NUM_SAMPLES=5000
SEED=42
RESULTS_BASE_DIR="./results"
DRY_RUN=false
FILTER=""
SAVE_EXAMPLES=true
NUM_EXAMPLES=20

# LAION Dataset Settings
LAION_DATA_ROOT="/mnt/lustre/datasets/laion400m/laion400m-data"
LAION_JSON_ROOT="/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m-negatives-with-swaps/"
TAR_RANGE="0,10"

# Multi-GPU settings
NUM_GPUS=1  # Number of GPUs to use in parallel (1 = sequential)

# SLURM settings
USE_SLURM=false
SLURM_PARTITION="gpu"
SLURM_GPUS=1
SLURM_TIME="02:00:00"
SLURM_MEM="32G"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num_samples|-n)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        --dry-run|--dry)
            DRY_RUN=true
            shift
            ;;
        --filter|-f)
            FILTER="$2"
            shift 2
            ;;
        --slurm)
            USE_SLURM=true
            shift
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
        --tar_range)
            TAR_RANGE="$2"
            shift 2
            ;;
        --num_gpus|-g)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config FILE         Path to eval_checkpoints.yaml (default: configs/eval_checkpoints.yaml)"
            echo "  --dataset NAME        Dataset to use: coco or laion (default: laion)"
            echo "  --num_samples N       Number of samples per model (default: 5000)"
            echo "  --results_dir DIR     Base directory for results (default: ./results)"
            echo "  --dry-run             Show commands without running"
            echo "  --filter PATTERN      Only run checkpoints matching pattern"
            echo "  --slurm               Submit jobs to SLURM"
            echo ""
            echo "COCO Options:"
            echo "  --coco_json PATH      Path to COCO JSON folder"
            echo "  --image_root PATH     Path to COCO image root"
            echo ""
            echo "LAION Options:"
            echo "  --laion_data_root     Path to LAION tar files"
            echo "  --laion_json_root     Path to LAION JSON negatives folder"
            echo "  --tar_range           Range of tar files to use (e.g., '0,10')"
            echo ""
            echo "Multi-GPU Options:"
            echo "  --num_gpus N          Number of GPUs to use in parallel (default: 1 = sequential)"
            echo "                        Each checkpoint runs on a single GPU"
            echo ""
            echo "  --help                Show this help"
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
    echo "❌ Python not found"
    exit 1
fi

if ! python -c "import yaml" &> /dev/null; then
    echo "❌ PyYAML not installed. Run: pip install pyyaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# =============================================================================
# PARSE YAML AND RUN EXPERIMENTS
# =============================================================================

echo "============================================================"
echo "Batch Half-Truth Vulnerability Experiment"
echo "============================================================"
echo "Config file:    $CONFIG_FILE"
echo "Dataset:        $DATASET"
echo "Num samples:    $NUM_SAMPLES"
echo "Results dir:    $RESULTS_BASE_DIR"
echo "Num GPUs:       $NUM_GPUS"
echo "Dry run:        $DRY_RUN"
if [ -n "$FILTER" ]; then
    echo "Filter:         $FILTER"
fi
if [ "$DATASET" = "coco" ]; then
    echo "COCO JSON:      $COCO_JSON"
    echo "Image root:     $COCO_IMAGE_ROOT"
elif [ "$DATASET" = "laion" ]; then
    echo "LAION data:     $LAION_DATA_ROOT"
    echo "LAION JSON:     $LAION_JSON_ROOT"
    echo "Tar range:      $TAR_RANGE"
fi
echo "============================================================"
echo ""

# Export environment variables for Python script
export CONFIG_FILE DATASET COCO_JSON COCO_IMAGE_ROOT NUM_SAMPLES SEED
export RESULTS_BASE_DIR DRY_RUN FILTER USE_SLURM SLURM_PARTITION
export SLURM_GPUS SLURM_TIME SLURM_MEM SAVE_EXAMPLES NUM_EXAMPLES
export LAION_DATA_ROOT LAION_JSON_ROOT TAR_RANGE NUM_GPUS

# Use Python to parse YAML and generate commands
python << 'PYTHON_SCRIPT'
import yaml
import os
import subprocess
import sys

# Read configuration from environment/arguments
config_file = os.environ.get('CONFIG_FILE', 'configs/eval_checkpoints.yaml')
dataset = os.environ.get('DATASET', 'coco')
coco_json = os.environ.get('COCO_JSON', 'swap_pos_json/coco_val/')
coco_image_root = os.environ.get('COCO_IMAGE_ROOT', '.')
num_samples = os.environ.get('NUM_SAMPLES', '1000')
seed = os.environ.get('SEED', '42')
results_base_dir = os.environ.get('RESULTS_BASE_DIR', './results')
dry_run = os.environ.get('DRY_RUN', 'false').lower() == 'true'
filter_pattern = os.environ.get('FILTER', '')
use_slurm = os.environ.get('USE_SLURM', 'false').lower() == 'true'
slurm_partition = os.environ.get('SLURM_PARTITION', 'gpu')
slurm_gpus = os.environ.get('SLURM_GPUS', '1')
slurm_time = os.environ.get('SLURM_TIME', '02:00:00')
slurm_mem = os.environ.get('SLURM_MEM', '32G')
save_examples = os.environ.get('SAVE_EXAMPLES', 'true').lower() == 'true'
num_examples = os.environ.get('NUM_EXAMPLES', '20')

# LAION-specific settings
laion_data_root = os.environ.get('LAION_DATA_ROOT', '/mnt/lustre/datasets/laion400m/laion400m-data')
laion_json_root = os.environ.get('LAION_JSON_ROOT', '/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m-negatives-with-swaps/')
tar_range = os.environ.get('TAR_RANGE', '0,10')

# Multi-GPU settings
num_gpus = int(os.environ.get('NUM_GPUS', '1'))

# Load YAML config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

checkpoints = config.get('checkpoints', [])

if not checkpoints:
    print("❌ No checkpoints found in config file")
    sys.exit(1)

print(f"📋 Found {len(checkpoints)} checkpoints in config")
if num_gpus > 1:
    print(f"🚀 Running in parallel on {num_gpus} GPUs")
print("")


def build_command(ckpt, idx):
    """Build the command for a checkpoint."""
    name = ckpt.get('name', f'checkpoint_{idx}')
    checkpoint_type = ckpt.get('checkpoint_type', 'openclip')
    checkpoint_path = ckpt.get('checkpoint_path', '')
    base_model = ckpt.get('base_model', 'ViT-B/32')
    csv_filename = ckpt.get('csv_filename', f'{name.replace(" ", "_")}.csv')
    force_openclip = ckpt.get('force_openclip', False)
    pretrained = ckpt.get('pretrained', 'openai')
    clove_weight = ckpt.get('clove_weight', 0.6)
    
    output_name = csv_filename.replace('.csv', '')
    output_dir = os.path.join(results_base_dir, f"half_truth_{dataset}_{output_name}")
    
    cmd = ["python", "experiments/half_truth_vulnerability.py"]
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--model_name", base_model])
    cmd.extend(["--checkpoint_type", checkpoint_type])
    
    if checkpoint_path:
        cmd.extend(["--checkpoint_path", checkpoint_path])
    
    cmd.extend(["--pretrained", pretrained])
    
    if force_openclip:
        cmd.append("--force_openclip")
    
    if checkpoint_type == "clove":
        cmd.extend(["--clove_weight", str(clove_weight)])
    
    if dataset == "coco":
        cmd.extend(["--json_folder", coco_json])
        cmd.extend(["--image_root", coco_image_root])
    elif dataset == "laion":
        cmd.extend(["--laion_data_root", laion_data_root])
        cmd.extend(["--laion_json_root", laion_json_root])
        cmd.extend(["--tar_range", tar_range])
    
    cmd.extend(["--num_samples", num_samples])
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--seed", seed])
    
    if save_examples:
        cmd.append("--save_examples")
        cmd.extend(["--num_examples", num_examples])
    
    return cmd, output_dir, name


def run_single_checkpoint(args):
    """Run a single checkpoint with assigned GPU."""
    idx, ckpt, gpu_id = args
    cmd, output_dir, name = build_command(ckpt, idx)
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd_str = " ".join(cmd)
    print(f"[GPU {gpu_id}] � Starting: {name}")
    print(f"[GPU {gpu_id}]    Command: {cmd_str[:100]}...")
    
    if dry_run:
        print(f"[GPU {gpu_id}]    [DRY RUN - not executing]")
        return (name, True, "", output_dir)
    
    try:
        if use_slurm:
            slurm_cmd = [
                "srun",
                f"--partition={slurm_partition}",
                f"--gpus={slurm_gpus}",
                f"--time={slurm_time}",
                f"--mem={slurm_mem}",
            ] + cmd
            result = subprocess.run(slurm_cmd, check=True, env=env)
        else:
            result = subprocess.run(cmd, check=True, env=env)
        
        print(f"[GPU {gpu_id}] ✅ Completed: {name}")
        return (name, True, "", output_dir)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr[:500] if e.stderr else 'N/A'
        print(f"[GPU {gpu_id}] ❌ Failed: {name}")
        return (name, False, f"Exit code {e.returncode}: {stderr}", output_dir)
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ Failed: {name}")
        return (name, False, str(e), output_dir)


# Filter checkpoints
filtered_checkpoints = []
skipped = []

for i, ckpt in enumerate(checkpoints):
    name = ckpt.get('name', f'checkpoint_{i}')
    if filter_pattern and filter_pattern.lower() not in name.lower():
        skipped.append(name)
    else:
        filtered_checkpoints.append((i, ckpt))

print(f"📊 Will run {len(filtered_checkpoints)} checkpoints" + 
      (f" (skipped {len(skipped)} due to filter)" if skipped else ""))
print("")

# Track results
successful = []
failed = []

if num_gpus > 1 and len(filtered_checkpoints) > 1 and not use_slurm:
    # Parallel execution across multiple GPUs
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    print(f"🔄 Starting parallel execution on {num_gpus} GPUs...")
    print("="*60)
    
    # Prepare tasks with round-robin GPU assignment
    tasks = []
    for task_idx, (ckpt_idx, ckpt) in enumerate(filtered_checkpoints):
        gpu_id = task_idx % num_gpus
        tasks.append((ckpt_idx, ckpt, gpu_id))
    
    # Run with at most num_gpus concurrent jobs
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(run_single_checkpoint, task): task for task in tasks}
        
        for future in as_completed(futures):
            name, success, error, output_dir = future.result()
            if success:
                successful.append((name, output_dir))
            else:
                failed.append((name, error))
else:
    # Sequential execution
    for task_idx, (ckpt_idx, ckpt) in enumerate(filtered_checkpoints):
        name = ckpt.get('name', f'checkpoint_{ckpt_idx}')
        
        print(f"{'='*60}")
        print(f"[{task_idx+1}/{len(filtered_checkpoints)}] {name}")
        print(f"{'='*60}")
        
        gpu_id = task_idx % num_gpus if num_gpus > 1 else 0
        result_name, success, error, output_dir = run_single_checkpoint((ckpt_idx, ckpt, gpu_id))
        
        if success:
            successful.append((result_name, output_dir))
            if not dry_run:
                print(f"   Results saved to: {output_dir}")
        else:
            failed.append((result_name, error))
        
        print("")

# Print summary
print("")
print("="*60)
print("BATCH EXECUTION SUMMARY")
print("="*60)
print(f"✅ Successful: {len(successful)}")
for name, output_dir in successful:
    print(f"   • {name}")

if skipped:
    print(f"\n⏭️  Skipped (filtered): {len(skipped)}")
    for name in skipped:
        print(f"   • {name}")

if failed:
    print(f"\n❌ Failed: {len(failed)}")
    for name, error in failed:
        error_short = error[:100] + "..." if len(error) > 100 else error
        print(f"   • {name}: {error_short}")

print("="*60)

PYTHON_SCRIPT
