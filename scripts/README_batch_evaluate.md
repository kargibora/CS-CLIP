# Batch Checkpoint Evaluation Script

This script allows you to evaluate multiple checkpoints (local saved models, HuggingFace models, or OpenAI-OpenCLIP models) in a single batch run. Each checkpoint will be evaluated on benchmark datasets and results will be saved to separate CSV files.

## Features

- ✅ **Multiple checkpoint types supported:**
  - Local `.pt` checkpoint files (fine-tuned or LabCLIP/HNB)
  - HuggingFace Hub models
  - OpenAI-OpenCLIP models
  
- ✅ **Automatic checkpoint type detection** (fine-tuned vs feature-based)
- ✅ **Per-checkpoint dataset selection**
- ✅ **Separate CSV output for each checkpoint**
- ✅ **Comprehensive logging and error handling**
- ✅ **Memory management** (clears CUDA cache between evaluations)

## Usage

### 1. Edit the Configuration

Open `scripts/batch_evaluate_checkpoints.py` and modify the `CHECKPOINTS_TO_EVALUATE` list at the bottom of the file:

```python
CHECKPOINTS_TO_EVALUATE = [
    CheckpointConfig(
        name="My Model Name",              # Display name for logs
        csv_filename="output.csv",          # CSV filename (saved to evaluation_results/)
        checkpoint_type="local",            # 'local', 'huggingface', or 'openclip'
        checkpoint_path="/path/to/model.pt", # Path or model identifier
        base_model="ViT-B/32",             # Base CLIP model
        is_finetuned=True,                 # True/False/None (auto-detect)
        datasets=None,                      # None = all default datasets, or list of specific ones
    ),
    # Add more checkpoints...
]
```

### 2. Run the Script

```bash
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally
python scripts/batch_evaluate_checkpoints.py
```

### 3. Results

Results will be saved to `evaluation_results/` directory with the CSV filenames you specified.

## Example Configurations

### Example 1: Zero-shot OpenAI CLIP

```python
CheckpointConfig(
    name="OpenAI CLIP ViT-B/32 (Zero-shot)",
    csv_filename="baseline_openai_vitb32.csv",
    checkpoint_type="openclip",
    checkpoint_path="ViT-B/32",
    base_model="ViT-B/32",
)
```

### Example 2: OpenCLIP with Specific Weights

```python
CheckpointConfig(
    name="OpenCLIP ViT-L/14 OpenAI weights",
    csv_filename="openclip_vitl14_openai.csv",
    checkpoint_type="openclip",
    checkpoint_path="ViT-L-14@openai",
    base_model="ViT-L/14",
)
```

Available OpenCLIP models:
- `ViT-B/32`, `ViT-B/16`, `ViT-L/14`
- With pretrained weights: `ViT-L-14@openai`, `ViT-B-32@laion2b_s34b_b79k`, etc.

### Example 3: HuggingFace Hub Model

```python
CheckpointConfig(
    name="NegCLIP from HuggingFace",
    csv_filename="negclip_huggingface.csv",
    checkpoint_type="huggingface",
    checkpoint_path="meronym/negclip-base",
    base_model="ViT-B/32",
)
```

You can use any HuggingFace Hub model ID (e.g., `username/model-name`).

### Example 4: Local Fine-Tuned Checkpoint

```python
CheckpointConfig(
    name="My Fine-Tuned Model",
    csv_filename="my_finetuned_model.csv",
    checkpoint_type="local",
    checkpoint_path="/path/to/checkpoint_epoch_10.pt",
    base_model="ViT-B/32",
    is_finetuned=True,  # Explicitly set to True for FT models
)
```

### Example 5: Local LabCLIP/HNB Checkpoint (Head Only)

```python
CheckpointConfig(
    name="My LabCLIP Head Model",
    csv_filename="labclip_head_model.csv",
    checkpoint_type="local",
    checkpoint_path="/path/to/head_checkpoint.pt",
    base_model="ViT-B/32",
    is_finetuned=False,  # Head-only model
)
```

### Example 6: Evaluate Specific Datasets Only

```python
CheckpointConfig(
    name="Model on VALSE + Winoground only",
    csv_filename="specific_datasets.csv",
    checkpoint_type="openclip",
    checkpoint_path="ViT-B/16",
    base_model="ViT-B/16",
    datasets=['VALSE', 'Winoground', 'SugarCrepe'],  # Only these datasets
)
```

Default datasets (when `datasets=None`):
- VALSE
- Winoground
- SugarCrepe
- VL_CheckList
- COCO_Order
- VG_Relation
- VG_Attribution
- ColorSwap
- SVO_Probes

## Parameters Explanation

### CheckpointConfig Parameters

- **`name`** (str): Display name for this checkpoint (used in logs)

- **`csv_filename`** (str): Output CSV filename (will be saved to `evaluation_results/`)

- **`checkpoint_type`** (str): Type of checkpoint
  - `"local"`: Local `.pt` file
  - `"huggingface"`: HuggingFace Hub model
  - `"openclip"`: OpenAI-OpenCLIP model

- **`checkpoint_path`** (str): Path or identifier for the checkpoint
  - For `local`: Full path to `.pt` file (e.g., `/path/to/model.pt`)
  - For `huggingface`: HuggingFace Hub ID (e.g., `username/model-name`)
  - For `openclip`: Model name (e.g., `ViT-B/32` or `ViT-L-14@openai`)

- **`base_model`** (str): Base CLIP architecture
  - Examples: `"ViT-B/32"`, `"ViT-B/16"`, `"ViT-L/14"`
  - Used mainly for local checkpoints

- **`is_finetuned`** (bool or None): Whether model is fine-tuned
  - `True`: Fine-tuned model (full encoder + head)
  - `False`: Feature-based model (head only, like LabCLIP/HNB)
  - `None`: Auto-detect from checkpoint structure

- **`datasets`** (list or None): Datasets to evaluate
  - `None`: Use all default datasets
  - List of dataset names: Only evaluate specified datasets

- **`config_overrides`** (dict or None): Optional config overrides (advanced)

## Output

### Console Output

The script provides detailed logging:
- Loading progress for each checkpoint
- Evaluation progress for each dataset
- Summary of results for each checkpoint
- Final summary of all evaluations (success/failed)

### CSV Files

Each checkpoint gets its own CSV file in `evaluation_results/` with:
- Dataset names
- Metrics (accuracy, etc.)
- Scores for each benchmark

## Tips

### Memory Management

- The script automatically clears CUDA cache between checkpoints
- If running out of memory, evaluate fewer checkpoints at a time

### Checkpoint Auto-Detection

- Set `is_finetuned=None` to let the script auto-detect checkpoint type
- Auto-detection checks:
  - Config metadata in checkpoint file
  - State dict keys (encoder params vs head-only params)

### Custom Datasets

To evaluate only specific datasets, use the `datasets` parameter:

```python
datasets=['VALSE', 'Winoground']  # Only these two
```

### Parallel Evaluation

Currently, checkpoints are evaluated sequentially. For parallel evaluation, you could:
1. Create multiple copies of the script with different checkpoint lists
2. Run them in separate processes/terminals
3. Make sure each uses a different GPU (set `CUDA_VISIBLE_DEVICES`)

## Troubleshooting

### Error: "Checkpoint not found"

- Check that the path in `checkpoint_path` is correct
- Use absolute paths for local checkpoints

### Error: "Failed to load HuggingFace model"

- Verify the model ID exists on HuggingFace Hub
- Check internet connection (downloads model if not cached)
- Ensure you have proper authentication (if model is private)

### Error: "CUDA out of memory"

- Reduce batch size in config
- Evaluate fewer datasets at once
- Clear CUDA cache: `torch.cuda.empty_cache()`
- Use a smaller model

### Mixed Results in CSV

If you see unexpected results:
- Check `is_finetuned` parameter matches your checkpoint type
- Verify `base_model` matches the checkpoint's architecture
- Check console logs for warnings during loading

## Advanced Usage

### Custom Config Overrides

You can pass custom config overrides:

```python
CheckpointConfig(
    name="My Model",
    csv_filename="output.csv",
    checkpoint_type="local",
    checkpoint_path="/path/to/model.pt",
    base_model="ViT-B/32",
    config_overrides={
        'evaluation': {
            'batch_size': 32,
        },
        'model': {
            'embedding_dim': 512,
        }
    }
)
```

### Changing Output Directory

Edit the `main()` function:

```python
def main():
    evaluator = BatchCheckpointEvaluator(
        checkpoints=CHECKPOINTS_TO_EVALUATE,
        output_dir="my_custom_output_dir",  # Change this
        base_config_path="configs/hnb_alignment.yaml",
    )
```

## Support

For issues or questions, check the main project documentation or contact the maintainers.
