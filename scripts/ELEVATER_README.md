# ELEVATER Benchmark Evaluation

This directory contains scripts for evaluating models on the [ELEVATER Image Classification Benchmark](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC).

## Quick Start

### 1. Basic Usage (Command Line)

```bash
# Evaluate OpenCLIP ViT-B/32 baseline
python scripts/evaluate_elevater.py \
    --checkpoint_type openclip \
    --checkpoint_path "ViT-B/32" \
    --name "CLIP ViT-B/32" \
    --output_dir ./elevater_results

# Evaluate on specific datasets only
python scripts/evaluate_elevater.py \
    --checkpoint_type openclip \
    --checkpoint_path "ViT-B/32" \
    --datasets cifar10 cifar100 caltech101 \
    --output_dir ./elevater_results
```

### 2. Using Config File

```bash
python scripts/evaluate_elevater.py --config configs/elevater_eval.yaml
```

### 3. Evaluate Your Trained Models

```bash
# HuggingFace model
python scripts/evaluate_elevater.py \
    --checkpoint_type huggingface \
    --checkpoint_path "username/model-name" \
    --name "My Fine-tuned Model" \
    --output_dir ./elevater_results

# Local checkpoint
python scripts/evaluate_elevater.py \
    --checkpoint_type local \
    --checkpoint_path "/path/to/checkpoint.pt" \
    --base_model "ViT-B/32" \
    --name "My Local Model" \
    --output_dir ./elevater_results

# TripletCLIP model
python scripts/evaluate_elevater.py \
    --checkpoint_type tripletclip \
    --checkpoint_path "TripletCLIP/CC12M_TripletCLIP_ViTB12" \
    --name "TripletCLIP" \
    --output_dir ./elevater_results
```

## Supported Checkpoint Types

| Type | Description | Example |
|------|-------------|---------|
| `openclip` | Standard OpenCLIP models | `ViT-B/32`, `ViT-L-14@openai` |
| `huggingface` | HuggingFace Hub models | `username/model-name` |
| `tripletclip` | TripletCLIP (separate encoders) | `TripletCLIP/CC12M_TripletCLIP_ViTB12` |
| `local` | Local checkpoint with config | `/path/to/checkpoint.pt` |
| `external` | External checkpoint (state dict only) | `/path/to/weights.pt` |
| `projection` | CLIP + text projection layer | `/path/to/projection_checkpoint.pt` |
| `clove` | CLOVE with weight interpolation | `/path/to/clove_weights.pt` |

## ELEVATER Datasets

The benchmark includes 20 image classification datasets:

| Dataset | Classes | Test Size |
|---------|---------|-----------|
| `caltech101` | 102 | 6,084 |
| `cifar10` | 10 | 10,000 |
| `cifar100` | 100 | 10,000 |
| `country211` | 211 | 21,100 |
| `dtd` | 47 | 1,880 |
| `eurosat` | 10 | 5,400 |
| `fer2013` | 7 | 3,589 |
| `fgvc_aircraft` | 100 | 3,333 |
| `food101` | 101 | 25,250 |
| `gtsrb` | 43 | 12,630 |
| `hateful_memes` | 2 | 500 |
| `kitti_distance` | 4 | 711 |
| `mnist` | 10 | 10,000 |
| `flowers102` | 102 | 6,149 |
| `oxford_pets` | 37 | 3,669 |
| `patch_camelyon` | 2 | 32,768 |
| `sst2` | 2 | 1,821 |
| `resisc45` | 45 | 6,300 |
| `stanford_cars` | 196 | 8,041 |
| `voc2007` | 20 | 4,952 |

## Configuration File Format

Create a YAML config file (e.g., `configs/elevater_eval.yaml`):

```yaml
# Model identification
name: "My Model"

# Checkpoint configuration
checkpoint_type: huggingface
checkpoint_path: "username/model-name"
base_model: "ViT-B/32"

# Output settings
output_dir: "./elevater_results"

# Datasets (optional - defaults to all 20)
datasets:
  - cifar10
  - cifar100
  - caltech101

# Evaluation settings
use_fp32: true
batch_size: 64
num_workers: 4
```

## Output Format

Results are saved as JSON files:

```json
{
  "model_name": "My Model",
  "checkpoint_type": "huggingface",
  "checkpoint_path": "username/model-name",
  "results": {
    "cifar10": 91.5,
    "cifar100": 72.3,
    "caltech101": 87.2
  },
  "average": 83.67,
  "num_datasets": 3
}
```

## Optional: Full ELEVATER Toolkit

For the complete ELEVATER experience with all 20 datasets, install the toolkit:

```bash
# Clone ELEVATER
git clone https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC.git
cd Elevater_Toolkit_IC

# Install
pip install -e .
```

The script will automatically detect and use the ELEVATER toolkit if installed, otherwise it falls back to a standalone implementation using torchvision datasets.

## Command-Line Options

```
usage: evaluate_elevater.py [-h] [--config CONFIG]
                            [--checkpoint_type {openclip,huggingface,local,tripletclip,external,projection,clove}]
                            [--checkpoint_path CHECKPOINT_PATH]
                            [--base_model BASE_MODEL]
                            [--name NAME]
                            [--output_dir OUTPUT_DIR]
                            [--datasets DATASETS [DATASETS ...]]
                            [--batch_size BATCH_SIZE]
                            [--num_workers NUM_WORKERS]
                            [--use_fp16]
                            [--force_openclip]
                            [--projection_path PROJECTION_PATH]
                            [--clove_weight CLOVE_WEIGHT]
                            [--pretrained PRETRAINED]

Options:
  --config, -c          Path to YAML config file
  --checkpoint_type     Type of checkpoint (default: openclip)
  --checkpoint_path     Path or identifier for checkpoint
  --base_model          Base CLIP architecture (default: ViT-B/32)
  --name                Model name for output files
  --output_dir          Output directory (default: ./elevater_results)
  --datasets            List of datasets to evaluate
  --batch_size          Batch size (default: 64)
  --num_workers         Data loading workers (default: 4)
  --use_fp16            Use FP16 instead of FP32
  --force_openclip      Force using OpenCLIP for loading
  --projection_path     Path to projection layer checkpoint
  --clove_weight        Weight for CLOVE interpolation (default: 0.6)
  --pretrained          Pretrained weights for base model
```

## Examples

### Evaluate Multiple Models

```bash
# Baseline CLIP models
for model in "ViT-B/32" "ViT-B/16" "ViT-L/14"; do
    python scripts/evaluate_elevater.py \
        --checkpoint_type openclip \
        --checkpoint_path "$model" \
        --name "CLIP $model" \
        --output_dir ./elevater_results
done
```

### Compare with NegCLIP

```bash
# NegCLIP
python scripts/evaluate_elevater.py \
    --checkpoint_type huggingface \
    --checkpoint_path "timbrooks/neg-clip" \
    --name "NegCLIP" \
    --output_dir ./elevater_results
```

### Batch Evaluation Script

```bash
#!/bin/bash
# batch_elevater_eval.sh

MODELS=(
    "openclip:ViT-B/32:CLIP ViT-B/32"
    "huggingface:timbrooks/neg-clip:NegCLIP"
    "tripletclip:TripletCLIP/CC12M_TripletCLIP_ViTB12:TripletCLIP"
)

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r type path name <<< "$model_spec"
    python scripts/evaluate_elevater.py \
        --checkpoint_type "$type" \
        --checkpoint_path "$path" \
        --name "$name" \
        --output_dir ./elevater_results
done
```

## Related Scripts

- `batch_evaluate_checkpoints.py` - Evaluate on compositional benchmarks (VALSE, Winoground, etc.)
- `test_clip_benchmark.py` - Quick CLIP benchmark tests
