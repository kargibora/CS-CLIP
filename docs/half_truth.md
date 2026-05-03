# Half-Truth Evaluation

The main training and checkpoint evaluation flow does not depend on the half-truth scripts.

These scripts are kept for diagnostic evaluation:

- `run_half_truth.sh`
- `run_half_truth_batch.sh`
- `run_image_half_truth_batch.sh`

## Caption-Side Half-Truth

`run_half_truth.sh` evaluates whether a model prefers longer partially incorrect captions over shorter correct captions.

Required COCO inputs:

```bash
COCO_JSON=/path/to/coco_val_entities
COCO_IMAGE_ROOT=/path/to/image_root
```

Example:

```bash
COCO_JSON=/path/to/coco_val_entities \
COCO_IMAGE_ROOT=/path/to/image_root \
./run_half_truth.sh \
  --checkpoint_path /path/to/checkpoint.pt \
  --checkpoint_type external
```

## Batch Caption-Side Half-Truth

`run_half_truth_batch.sh` reads checkpoints from `configs/eval_checkpoints.yaml`.

Example:

```bash
COCO_JSON=/path/to/coco_val_entities \
COCO_IMAGE_ROOT=/path/to/image_root \
./run_half_truth_batch.sh --config configs/eval_checkpoints.yaml
```

## Image-Side Half-Truth

`run_image_half_truth_batch.sh` evaluates image retrieval under partially shared visual content.

Example:

```bash
COCO_JSON=/path/to/coco_val_entities \
COCO_IMAGE_ROOT=/path/to/image_root \
./run_image_half_truth_batch.sh --config configs/eval_checkpoints.yaml
```
