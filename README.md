# CS-CLIP

Code for full CLIP fine-tuning with structured entity and relation negatives.

The public pipeline has two steps:

1. Train CLIP using prepared entity/relation JSON files.
2. Evaluate saved checkpoints on installed benchmark datasets.

The LLM data-generation pipeline is not required for training or evaluation.

## Setup

Create and activate your Python environment, then install the dependencies.

```bash
pip install -r requirements.txt
```

## Data

Training needs:

- COCO images, or another image folder referenced by your JSON files.
- Entity/relation JSON files in the expected format.

Evaluation needs:

- The benchmark datasets you want to evaluate on.
- A dataset root passed with `EVAL_DATA_ROOT`.

No dataset or checkpoint paths are hard-coded. The scripts read paths from environment variables.

## COCO Images

If your training JSON `image_path` values look like `datasets/COCO/train2014/...`, use `IMAGE_ROOT=.`.

One possible COCO layout is:

```text
datasets/
  COCO/
    train2014/
    val2014/
    val2017/
```

Example COCO download commands:

```bash
mkdir -p datasets/COCO
cd datasets/COCO

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2014.zip
unzip val2014.zip
unzip val2017.zip
```

## Training JSON Format

Each JSON sample should contain:

```json
{
  "sample_id": "example_id",
  "original_caption": "A person riding a horse.",
  "entities": ["person", "horse"],
  "negative_entities": {
    "horse": [
      {"negative": "cow", "change_type": "object_change"}
    ]
  },
  "relations": [
    {
      "subject": "person",
      "relation_type": "riding",
      "object": "horse",
      "negatives": [
        {
          "subject": "person",
          "relation_type": "standing next to",
          "object": "horse",
          "change_type": "predicate_change"
        }
      ]
    }
  ],
  "image_path": "datasets/COCO/train2014/COCO_train2014_000000000001.jpg"
}
```

The training script reads all `*.json` files from `TRAIN_JSON_DIR`.

## Train

Required variables:

- `RUN_NAME`
- `TRAIN_JSON_DIR`
- `IMAGE_ROOT`

Example:

```bash
RUN_NAME=coco_ft \
TRAIN_JSON_DIR=swap_pos_json/coco_train_entities \
IMAGE_ROOT=. \
GPUS=1 \
BATCH_SIZE=16 \
LR=5e-6 \
EPOCHS=5 \
SAVE_EVERY_K_STEPS=1000 \
./train_structured.sh
```

Checkpoints are written to:

```text
checkpoints/<RUN_NAME>/
```

## Evaluation Datasets

Evaluation datasets are resolved under `EVAL_DATA_ROOT`.

For example:

```text
datasets/
  WhatsUp/
  Winoground/
  SugarCrepe/
  VALSE/
```

Common dataset names:

- `VG_Attribution`, `VG_Relation`, `COCO_Order`, `Flickr30k_Order` use `EVAL_DATA_ROOT/WhatsUp`.
- `Winoground` uses `EVAL_DATA_ROOT/Winoground`.
- `SugarCrepe` uses `EVAL_DATA_ROOT/SugarCrepe` and expects COCO `val2017` inside that folder.
- `VALSE` uses `EVAL_DATA_ROOT/VALSE` as its Hugging Face cache directory.

Some loaders download from Hugging Face on first use. Others expect files to already exist locally.

## Evaluate

Required variables:

- `CHECKPOINT_PATH`
- `CHECKPOINT_CONFIG`

Optional variables:

- `EVAL_DATA_ROOT`
- `DATASETS`
- `OUTPUT_CSV`
- `OUTPUT_DIR`
- `BASE_MODEL`

Example:

```bash
CHECKPOINT_PATH=checkpoints/coco_ft/checkpoint_step_1000.pt \
CHECKPOINT_CONFIG=checkpoints/coco_ft/config.json \
EVAL_DATA_ROOT=datasets \
DATASETS=VG_Attribution \
OUTPUT_CSV=coco_ft_vg_attr.csv \
./eval_checkpoint.sh
```

Results are written to:

```text
evaluation_results/
```

## Half-Truth Evaluation

The half-truth scripts are documented separately in [docs/half_truth.md](docs/half_truth.md).
