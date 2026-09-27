# CS-CLIP: Component-Supervised CLIP

**[Half-Truths Break Similarity-Based Retrieval](https://arxiv.org/abs/2602.23906)** · NeurIPS 2026

Bora Kargi, Arnas Uselis, Seong Joon Oh

CS-CLIP fine-tunes CLIP with entity/relation units and matched foils while retaining standard dual-encoder inference. On the verified COCO–Qwen evaluation, Half-Truth accuracy is **34.0% for CLIP and 76.4% for CS-CLIP**. The compositional benchmark average improves by **5.8 points**.

## Install

Python 3.11 and a CUDA GPU are recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepared training data requires no language model, spaCy model, or benchmark downloads. Additional compositional benchmarks use `requirements-eval.txt`; generating new units/foils uses `requirements-generation.txt`.

## Datasets

- [CS-CLIP training annotations](https://huggingface.co/datasets/kbora/CS-CLIP-Training): 410,340 caption records, including units, matched foils, and shuffled full-caption negatives.
- [Half-Truths](https://huggingface.co/datasets/kbora/Half-Truths): 1,437 verified evaluation comparisons with images. Configurations: `coco_qwen` (509), `coco_mistral` (454), `cc3m_qwen` (474).

Download the exact training JSON files:

```bash
hf download kbora/CS-CLIP-Training original/training-json.tar.gz \
  --repo-type dataset --local-dir datasets/CS-CLIP-Training
mkdir -p datasets/CS-CLIP-Training/json
tar -xzf datasets/CS-CLIP-Training/original/training-json.tar.gz \
  -C datasets/CS-CLIP-Training/json
```

Download and extract [COCO train2014 images](http://images.cocodataset.org/zips/train2014.zip) to `datasets/COCO/train2014/`. The training loader accepts the archive's original `positive_components` / `negative_components` fields and the generator's `entities` / `negative_entities` fields.

## Train

Run from the repository root:

```bash
RUN_NAME=csclip \
TRAIN_JSON_DIR=datasets/CS-CLIP-Training/json \
IMAGE_ROOT=. \
GPUS=8 EPOCHS=25 \
./train_structured.sh
```

This command fine-tunes both encoders of OpenAI ViT-B/32 for 25 epochs, with batch size 128 per GPU, AdamW at `5e-6`, weight decay `0.01`, two unit/foil pairs per image, and unit-loss weight `0.5`.

Outputs are `checkpoints/csclip/last_checkpoint.pt`, `best_checkpoint.pt` when validation identifies one, and the full `config.json`. `last_checkpoint.pt` preserves the final training weights; the paper's evaluated weights are available below.

## Pre-trained checkpoints

| Model | Training images | Download |
|---|---|---|
| CS-CLIP ViT-B/32 | COCO | [Checkpoint](https://drive.google.com/file/d/14IBgBgKhCDhRHJfnDaFxsTo6ocoS4I6W/view) |

The linked checkpoint matches the evaluated paper checkpoint byte-for-byte (SHA-256: `556c2763469bb32a93d01732e1bf72d6bac5b4b194a7175421858e16beae4dcc`). Save it as `checkpoints/csclip/last_checkpoint.pt` for the command below.

## Evaluate Half-Truths

The evaluation command downloads the selected configuration from Hugging Face:

```bash
# Pretrained CLIP
python scripts/evaluate_half_truth.py --subset coco_qwen --output results/clip.json

# CS-CLIP
python scripts/evaluate_half_truth.py --subset coco_qwen \
  --checkpoint checkpoints/csclip/last_checkpoint.pt --output results/csclip.json
```

Use `coco_mistral` or `cc3m_qwen` for the other configurations. The output includes per-comparison scores, counts, entity/relation accuracy, and Correct-Ordering Rate. See [metric definitions](docs/half_truth.md).

| Model | COCO–Qwen | COCO–Mistral | CC3M–Qwen |
|---|---:|---:|---:|
| CLIP | 34.0 | 28.6 | 36.1 |
| CS-CLIP | 76.4 | 67.2 | 66.2 |

Values are Half-Truth accuracy (%). The released rows are evaluation-only.

## Optional workflows

For the compositional benchmarks, install `requirements-eval.txt`, prepare the relevant benchmark's data, and run:

```bash
CHECKPOINT_PATH=checkpoints/csclip/last_checkpoint.pt \
CHECKPOINT_CONFIG=checkpoints/csclip/config.json \
EVAL_DATA_ROOT=datasets DATASETS=VG_Attribution ./eval_checkpoint.sh
```

To generate units and foils for new captions, install `requirements-generation.txt` and run `python cli.py --help`. This is optional: the prepared training archive is the reproducible source for the reported run.

## Checks

```bash
python -m unittest discover -s tests -v
```

These check the unit-loss equation and gradients, original training-data schema, final/best checkpoint separation, complete configuration saving, and strict HT/COR comparisons.

## Citation

Code is released under the [MIT license](LICENSE). Generated annotations use CC BY 4.0; source images and captions retain their original terms.

```bibtex
@inproceedings{kargi2026halftruths,
  title={Half-Truths Break Similarity-Based Retrieval},
  author={Kargi, Bora and Uselis, Arnas and Oh, Seong Joon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2602.23906}
}
```
