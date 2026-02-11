# Batch Checkpoint Evaluation - Quick Start Guide

## Overview

The batch evaluation script allows you to evaluate multiple checkpoints in parallel across multiple GPUs. Each checkpoint will be evaluated on benchmark datasets and results saved to separate CSV files.

## Quick Start

### 1. Using YAML Configuration (Recommended)

Create or edit `configs/eval_checkpoints.yaml`:

```yaml
checkpoints:
  - name: "My Model Epoch 5"
    csv_filename: "my_model_epoch_5.csv"
    checkpoint_type: "local"
    checkpoint_path: "/path/to/checkpoint_epoch_5.pt"
    base_model: "ViT-B/32"
    is_finetuned: true
  
  - name: "My Model Epoch 10"
    csv_filename: "my_model_epoch_10.csv"
    checkpoint_type: "local"
    checkpoint_path: "/path/to/checkpoint_epoch_10.pt"
    base_model: "ViT-B/32"
    is_finetuned: true
```

Run evaluation:

```bash
# Single GPU
python scripts/batch_evaluate_checkpoints.py --config configs/eval_checkpoints.yaml

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 scripts/batch_evaluate_checkpoints.py \
    --config configs/eval_checkpoints.yaml
```

### 2. Using Command Line (Single Checkpoint)

```bash
python scripts/batch_evaluate_checkpoints.py \
    --checkpoint_type local \
    --checkpoint_path "/path/to/checkpoint.pt" \
    --csv_filename "my_results.csv" \
    --name "My Model" \
    --base_model "ViT-B/32" \
    --is_finetuned true
```

## Multi-GPU Evaluation

The script distributes checkpoints across GPUs using round-robin assignment:

```bash
# 8 GPUs - each GPU evaluates different checkpoints
torchrun --nproc_per_node=8 scripts/batch_evaluate_checkpoints.py \
    --config configs/eval_checkpoints.yaml
```

**Example distribution for 10 checkpoints on 8 GPUs:**
- GPU 0: Checkpoints 0, 8
- GPU 1: Checkpoint 1, 9
- GPU 2: Checkpoint 2
- GPU 3: Checkpoint 3
- GPU 4: Checkpoint 4
- GPU 5: Checkpoint 5
- GPU 6: Checkpoint 6
- GPU 7: Checkpoint 7

## Checkpoint Types

### 1. Local Checkpoints (.pt files)

```yaml
- name: "My Fine-tuned Model"
  csv_filename: "finetuned.csv"
  checkpoint_type: "local"
  checkpoint_path: "/absolute/path/to/checkpoint.pt"
  base_model: "ViT-B/32"
  is_finetuned: true  # or false for head-only models
```

### 2. HuggingFace Hub Models

```yaml
- name: "NegCLIP from HuggingFace"
  csv_filename: "negclip.csv"
  checkpoint_type: "huggingface"
  checkpoint_path: "meronym/negclip-base"
  base_model: "ViT-B/32"
```

### 3. OpenCLIP Models

```yaml
- name: "OpenAI CLIP Baseline"
  csv_filename: "baseline.csv"
  checkpoint_type: "openclip"
  checkpoint_path: "ViT-B/32"
  base_model: "ViT-B/32"
```

OpenCLIP with specific weights:
```yaml
- name: "OpenCLIP ViT-L/14"
  csv_filename: "vitl14.csv"
  checkpoint_type: "openclip"
  checkpoint_path: "ViT-L-14@openai"
  base_model: "ViT-L/14"
```

## Command-Line Arguments

```
--config PATH             Path to YAML config file with checkpoint list
--name NAME               Name for this checkpoint
--csv_filename FILE       Output CSV filename
--checkpoint_type TYPE    Type: local, huggingface, or openclip
--checkpoint_path PATH    Path or identifier for checkpoint
--base_model MODEL        Base CLIP model (default: ViT-B/32)
--is_finetuned BOOL       true/false/auto (default: auto)
--datasets LIST           Specific datasets to evaluate
--output_dir DIR          Output directory (default: evaluation_results)
--base_config PATH        Base config file (default: configs/hnb_alignment.yaml)
```

## Evaluating Specific Datasets

By default, all datasets are evaluated. To evaluate specific ones:

**In YAML:**
```yaml
- name: "My Model"
  csv_filename: "results.csv"
  checkpoint_type: "local"
  checkpoint_path: "/path/to/model.pt"
  base_model: "ViT-B/32"
  datasets:
    - "VALSE"
    - "Winoground"
    - "SugarCrepe"
```

**Command line:**
```bash
python scripts/batch_evaluate_checkpoints.py \
    --checkpoint_type openclip \
    --checkpoint_path "ViT-B/32" \
    --csv_filename results.csv \
    --datasets VALSE Winoground SugarCrepe
```

## Default Datasets

When `datasets` is not specified:
- VALSE
- Winoground
- SugarCrepe
- VL_CheckList
- COCO_Order
- VG_Relation
- VG_Attribution
- ColorSwap
- SVO_Probes

## Output

Results are saved to `evaluation_results/` (or `--output_dir`):
- One CSV file per checkpoint
- Each CSV contains all metrics for all datasets

## Example: Evaluating Multiple Training Checkpoints

```yaml
checkpoints:
  # Evaluate checkpoints from different epochs
  - name: "Training Epoch 5"
    csv_filename: "epoch_05.csv"
    checkpoint_type: "local"
    checkpoint_path: "/path/to/checkpoint_epoch_5.pt"
    base_model: "ViT-B/32"
    is_finetuned: true
  
  - name: "Training Epoch 10"
    csv_filename: "epoch_10.csv"
    checkpoint_type: "local"
    checkpoint_path: "/path/to/checkpoint_epoch_10.pt"
    base_model: "ViT-B/32"
    is_finetuned: true
  
  - name: "Training Epoch 15"
    csv_filename: "epoch_15.csv"
    checkpoint_type: "local"
    checkpoint_path: "/path/to/checkpoint_epoch_15.pt"
    base_model: "ViT-B/32"
    is_finetuned: true
  
  - name: "Best Model"
    csv_filename: "best.csv"
    checkpoint_type: "local"
    checkpoint_path: "/path/to/checkpoint_best.pt"
    base_model: "ViT-B/32"
    is_finetuned: true
  
  # Compare with baseline
  - name: "Zero-shot Baseline"
    csv_filename: "baseline.csv"
    checkpoint_type: "openclip"
    checkpoint_path: "ViT-B/32"
    base_model: "ViT-B/32"
```

Run with 8 GPUs:
```bash
torchrun --nproc_per_node=8 scripts/batch_evaluate_checkpoints.py \
    --config configs/eval_checkpoints.yaml \
    --output_dir my_evaluation_results
```

## Troubleshooting

### CUDA Out of Memory
- Reduce number of checkpoints evaluated simultaneously
- Use fewer GPUs: `--nproc_per_node=4`
- The script already clears CUDA cache between checkpoints

### File Not Found
- Use absolute paths for local checkpoints
- Verify checkpoint files exist

### Wrong Results
- Check `is_finetuned` parameter matches your checkpoint type
- Verify `base_model` matches checkpoint's architecture
- Check console logs for loading warnings

### Distributed Hanging
- Ensure all GPUs are available
- Check NCCL environment variables
- Try single GPU first to isolate issues

## Tips

1. **Test first**: Run with 1-2 checkpoints to verify everything works
2. **Use descriptive names**: Makes it easier to identify results later
3. **Organize output**: Use subdirectories in output_dir for different experiments
4. **Monitor logs**: Each GPU logs its assigned checkpoints
5. **CSV naming**: Use consistent naming scheme for easy sorting

## Example Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# Load modules
module load cuda/11.8
module load python/3.9

# Activate environment
source /path/to/venv/bin/activate

# Run evaluation
cd /path/to/CLIP-not-BoW-unimodally
torchrun --nproc_per_node=8 scripts/batch_evaluate_checkpoints.py \
    --config configs/eval_checkpoints.yaml \
    --output_dir results/$(date +%Y%m%d_%H%M%S)
```
