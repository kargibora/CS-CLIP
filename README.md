# CS-CLIP: Component-Supervised CLIP

**Half-Truths Break Similarity-Based Retrieval · NeurIPS 2026**
Bora Kargi, Arnas Uselis, Seong Joon Oh

[Paper](https://arxiv.org/abs/2602.23906) · [Project page](https://kargibora.github.io/half-truths/) · [Model](https://huggingface.co/kbora/CS-CLIP) · [Evaluation data](https://huggingface.co/datasets/kbora/Half-Truths) · [Training data](https://huggingface.co/datasets/kbora/CS-CLIP-Training)

CLIP can assign higher image–text similarity after a plausible but incorrect detail is added to a correct caption. We study this half-truth failure and introduce CS-CLIP, which fine-tunes CLIP with entity/relation units and matched foils while retaining standard dual-encoder inference.

## Use the model

The ViT-B/32 checkpoint loads directly with OpenCLIP:

```bash
pip install 'open_clip_torch>=2.24,<3'
```

```python
import open_clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = open_clip.create_model_from_pretrained(
    "hf-hub:kbora/CS-CLIP", device=device
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("image.jpg").convert("RGB")).unsqueeze(0).to(device)
captions = ["A dog runs on the grass.", "A dog runs on the grass beside a bicycle."]
with torch.inference_mode():
    image_features = model.encode_image(image, normalize=True)
    text_features = model.encode_text(tokenizer(captions).to(device), normalize=True)
    similarities = image_features @ text_features.T
print(similarities[0].cpu().tolist())
```

Replace `image.jpg` and the captions with your own inputs. Scores are cosine similarities. See the [model card](https://huggingface.co/kbora/CS-CLIP) for checkpoint details.

## Install the repository

Use Python 3.11. Training requires CUDA; evaluation also supports CPU.

```bash
git clone https://github.com/kargibora/CS-CLIP.git
cd CS-CLIP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluate Half-Truths

Download the checkpoint:

```bash
hf download kbora/CS-CLIP open_clip_pytorch_model.bin \
  --local-dir checkpoints/csclip
```

Evaluate CS-CLIP and the pretrained CLIP baseline:

```bash
python scripts/evaluate_half_truth.py --subset coco_qwen \
  --checkpoint checkpoints/csclip/open_clip_pytorch_model.bin \
  --output results/csclip-coco_qwen.json

python scripts/evaluate_half_truth.py --subset coco_qwen \
  --output results/clip-coco_qwen.json
```

The selected evaluation set, including its images, downloads automatically. Use `coco_mistral` or `cc3m_qwen` for the other sets. Add `--device cpu` for CPU evaluation.

| Model | COCO–Qwen | COCO–Mistral | CC3M–Qwen |
|---|---:|---:|---:|
| CLIP | 34.0 | 28.6 | 36.1 |
| CS-CLIP | **76.4** | **67.2** | **66.2** |

Half-Truth accuracy (%). CS-CLIP also achieves **57.8%** average image-to-text accuracy across the 16 compositional benchmarks.

Each output JSON contains overall and entity/relation results, Correct-Ordering Rate, sample counts, and per-comparison scores. See [evaluation details and offline usage](docs/half_truth.md).

## Datasets

| Dataset | Contents | Use |
|---|---|---|
| [Half-Truths](https://huggingface.co/datasets/kbora/Half-Truths) | 1,437 verified comparisons with images: COCO–Qwen (509), COCO–Mistral (454), CC3M–Qwen (474) | Evaluation |
| [CS-CLIP-Training](https://huggingface.co/datasets/kbora/CS-CLIP-Training) | 410,340 caption records with units, matched foils, and shuffled full-caption negatives | Training |

Training annotations reference COCO `train2014`; download those images separately. The original JSON archive is the training input. The Hugging Face Parquet view is provided for browsing.

## Train

Download and extract the annotations:

```bash
hf download kbora/CS-CLIP-Training original/training-json.tar.gz \
  --repo-type dataset --local-dir datasets/CS-CLIP-Training
mkdir -p datasets/CS-CLIP-Training/json
tar -xzf datasets/CS-CLIP-Training/original/training-json.tar.gz \
  -C datasets/CS-CLIP-Training/json
```

Download and extract [COCO train2014 images](http://images.cocodataset.org/zips/train2014.zip) to `datasets/COCO/train2014/`. Run from the repository root:

```bash
RUN_NAME=csclip \
TRAIN_JSON_DIR=datasets/CS-CLIP-Training/json \
IMAGE_ROOT=. \
GPUS=8 EPOCHS=25 \
./train_structured.sh
```

This command fine-tunes both encoders of OpenAI ViT-B/32 for 25 epochs, with batch size 128 per GPU, AdamW at `5e-6`, weight decay `0.01`, two unit/foil pairs per image, and unit-loss weight `0.5`.

Training saves `checkpoints/csclip/last_checkpoint.pt`, `best_checkpoint.pt` when validation identifies one, and the full `config.json`. The final checkpoints contain plain CLIP encoder state dictionaries and load directly with `model.load_state_dict(torch.load(path, weights_only=True))`. To evaluate your trained model, pass its checkpoint to `scripts/evaluate_half_truth.py`.

## Additional workflows

**Compositional benchmarks.** Install `requirements-eval.txt` and prepare the relevant benchmark's data. For a run saved by the training command above:

```bash
CHECKPOINT_PATH=checkpoints/csclip/last_checkpoint.pt \
CHECKPOINT_CONFIG=checkpoints/csclip/config.json \
EVAL_DATA_ROOT=datasets DATASETS=VG_Attribution ./eval_checkpoint.sh
```

**Generate annotations.** Install `requirements-generation.txt` and run `python cli.py --help`. The released training annotations can be used directly without running a language model.

**Run checks.**

```bash
python -m unittest discover -s tests -v
```

## License and citation

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
