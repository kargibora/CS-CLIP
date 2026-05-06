# CS-CLIP: Component-Supervised CLIP

When a text description is extended with an additional detail, image-text similarity should drop if that detail is wrong. We show that CLIP-style dual encoders often violate this intuition: appending a plausible but incorrect object or relation to an otherwise correct description can increase the similarity score. We call such cases *half-truths*. On COCO, CLIP prefers the correct shorter description only 40.6% of the time, and performance drops to 32.9% when the added detail is a relation. We trace this vulnerability to weak supervision on caption parts: contrastive training aligns full sentences but does not explicitly enforce that individual entities and relations are grounded. We propose CS-CLIP (Component-Supervised CLIP), which decomposes captions into entity and relation units, constructs a minimally edited foil for each unit, and fine-tunes the model to score the correct unit above its foil while preserving standard dual-encoder inference. CS-CLIP raises half-truth accuracy to 69.3% and improves average performance on established compositional benchmarks by 5.7 points, suggesting that reducing half-truth errors aligns with broader gains in compositional understanding.

## Release Status

- [x] CS-CLIP COCO Checkpoint
- [x] Pre-extracted negatives
- [x] Training and evaluation code
- [x] Unit extraction pipeline
- [ ] Dataset setup instructions

## Pre-trained Checkpoints

| Model | Dataset | Download |
|-------|---------|----------|
| CS-CLIP-ViT-B/32 | MSCOCO | [link](https://drive.google.com/file/d/14IBgBgKhCDhRHJfnDaFxsTo6ocoS4I6W/view?usp=drive_link) |

## Datasets

Pre-extracted caption units (entities and relations) used for CS-CLIP training.

| Dataset | Description | Download |
|--------|-------------|----------|
| MSCOCO | Caption samples with extracted entities and relations | [link](https://drive.google.com/file/d/1DpthIA-5zT_m1GKfqvHUWWH_z2XyEOMP/view?usp=drive_link) |

---

The public pipeline has three steps:

1. **(Optional)** Generate entity/relation unit JSON files from captions using the unit pipeline.
2. Train CLIP using prepared entity/relation JSON files.
3. Evaluate saved checkpoints on installed benchmark datasets.

Pre-extracted JSON files for MSCOCO are available for download above, so step 1 is only needed if you want to generate your own data.

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

## Unit Pipeline

The unit pipeline extracts entity and relation units from captions and generates minimally edited foils for each unit. It requires a GPU and [vLLM](https://github.com/vllm-project/vllm).

**Required arguments:**

| Argument | Description |
|---|---|
| `--coco_karpathy` | Path to the COCO Karpathy split JSON (`dataset_coco.json`) |
| `--coco_images_root` | Root directory containing COCO image folders (`train2014/`, `val2014/`, etc.) |
| `--output` | Output path for the structured negative JSON |

**Common optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--coco_split` | all splits | Filter to a specific split: `train`, `val`, `test`, or `restval` |
| `--subset START END` | all | Process only captions `[START, END)` — useful for testing |
| `--positives_output` | — | Also save a JSON of extracted positives (entities + relations, no foils) |
| `--llm_name` | `Qwen/Qwen3-14B-AWQ` | HuggingFace model name for the LLM |
| `--llm_batch` | `256` | Batch size passed to vLLM |
| `--n_neg_per_entity` | `2` | Number of foils to generate per entity unit |
| `--n_relational_negatives` | `3` | Number of foils to generate per relation unit |

**Example — smoke test on 5 captions:**

```bash
python cli.py \
  --coco_karpathy datasets/COCO/dataset_coco.json \
  --coco_images_root datasets/COCO \
  --coco_split val \
  --subset 0 5 \
  --output /tmp/test_out.json
```

**Example — full val split:**

```bash
python cli.py \
  --coco_karpathy datasets/COCO/dataset_coco.json \
  --coco_images_root datasets/COCO \
  --coco_split val \
  --output neg_json/coco_val.json \
  --positives_output pos_json/coco_val.json
```

The output JSON follows the training format described in the next section.

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


