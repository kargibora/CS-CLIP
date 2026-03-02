# Half-Truths Break Similarity-Based Retrieval

This repository contains the implementation for training CLIP with component-supervised contrastive learning. 

## TODO

Currently this repo is in WIP and will be updated for reproducability:
- [x] Release initial codebase
- [x] Release pre-trained checkpoints
- [ ] Release better documentation for dataset installation & pipeline usage
- [ ] Release pre-extracted negatives (COCO, LAION)
- [ ] Add evaluation scripts and benchmark reproduction

## Dataset Setup

Download your datasets and place them under the `datasets/` folder following each dataset's official guide:

```
datasets/
├── COCO/
│   ├── dataset_coco.json    # Karpathy split
│   ├── train2014/
│   └── val2014/
├── COCO_Order/
├── Flickr30k/
└── LAION400M/              # WebDataset format (.tar shards)
```

## Usage

### Step 1: Extract Components and Generate Negatives

Use `neg_pipeline` to extract visual components from captions and generate hard negatives:

```bash
# For COCO dataset with Karpathy split
python -m neg_pipeline.main \
    --coco_karpathy datasets/COCO/dataset_coco.json \
    --coco_images_root datasets/COCO \
    --coco_split train \
    --use_unified_generation \
    --unified_n_component_neg 2 \
    --unified_n_binding_pairs 2 \
    --unified_n_relational_neg 3 \
    --output neg_json/coco_negatives.json \
    --positives_output pos_json/coco_positives.json \
    --llm_name microsoft/Phi-3-mini-4k-instruct

# For WebDataset (e.g., LAION)
python -m neg_pipeline.main \
    --shards 'datasets/LAION400M/{00000..00100}.tar' \
    --use_unified_generation \
    --output neg_json/laion_negatives.json \
    --positives_output pos_json/laion_positives.json
```

**Key arguments:**
- `--use_unified_generation`: Fast mode that generates all negatives in a single LLM call
- `--use_relational_extraction`: Extract components with spatial/relational information
- `--use_component_negatives`: Generate negatives by modifying individual components
- `--use_relational_negatives`: Generate negatives by modifying spatial relations

### Step 2: Train the Model

Training uses [Hydra](https://hydra.cc/) for configuration management. Run `align.py` with a config:

```bash
# Train on COCO with component-aware loss
python align.py \
    --config-path=configs \
    --config-name=laion_ft \
    dataset=coco_order \
    training.epochs=5 \
    training.batch_size=128 \
    optimizer.learning_rate=5e-6

# Train on LAION400M (distributed)
torchrun --nproc_per_node=4 align.py \
    --config-path=configs \
    --config-name=laion_ft \
    dataset=laion400m \
    training.epochs=50 \
    dist.distributed=true
```

**Key training configurations:**
- `alignment.alignment_type`: `HNB` (pre-extracted features) or `FT` (end-to-end fine-tuning)
- `alignment.ft_image`: Fine-tune image encoder
- `alignment.ft_text`: Fine-tune text encoder
- `loss.lambda_component`: Weight for component loss

See `configs/` for available options.

## Pre-trained Checkpoints

Download our pre-trained checkpoints:

| Model | Dataset | Download |
|-------|---------|----------|
| CS-CLIP-ViT-B/32 | COCO | [link](https://drive.google.com/file/d/14IBgBgKhCDhRHJfnDaFxsTo6ocoS4I6W/view?usp=drive_link) |

**Loading a checkpoint:**

```python
import torch
from models import CLIPFeaturePipeline

# Load checkpoint
checkpoint = torch.load("checkpoints/your_checkpoint/best_checkpoint.pt")

# Initialize model and load weights
model = CLIPFeaturePipeline(...)  # See align.py for full initialization
model.load_state_dict(checkpoint)
```

## Project Structure

```
├── align.py                 # Main training script
├── neg_pipeline/            # Component extraction & negative generation
│   ├── main.py              # Entry point for neg_pipeline
│   ├── unified_generation.py
│   └── relational_generation.py
├── configs/                 # Hydra configuration files
│   ├── dataset/
│   ├── training/
│   ├── loss/
│   └── ...
├── models/                  # Model architectures
├── alignment/               # Training loop and losses
├── data_loading/            # Dataset implementations
└── checkpoints/             # Saved model checkpoints
```

## Citation

```bibtex
@article{kargi2026halftruths,
  title   = {Half-Truths Break Similarity-Based Retrieval},
  author  = {Kargi, Bora and Uselis, Arnas and Oh, Seong Joon},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.23906},
}
```


[MIT License](LICENSE)
