# Half-Truth evaluation

Install the repository's `requirements.txt`, then run:

```bash
python scripts/evaluate_half_truth.py --subset coco_qwen --output results/clip.json
python scripts/evaluate_half_truth.py --subset coco_qwen \
  --checkpoint checkpoints/csclip/last_checkpoint.pt --output results/csclip.json
```

The [Hugging Face release](https://huggingface.co/datasets/kbora/Half-Truths) provides `coco_qwen` (509 comparisons), `coco_mistral` (454), and `cc3m_qwen` (474). Each configuration is an evaluation `test` split. Images are embedded in Parquet; external image downloads are not needed.

For offline evaluation, pass `--data /path/to/coco_qwen.parquet`. A local JSONL file is also supported with `--image-root` resolving its `image` paths. Checkpoints must match the CLIP backbone exactly; incompatible weights fail loading.

For normalized embeddings, let `s` be cosine similarity:

- HT: `s(image, anchor) > s(image, half_truth)`.
- COR: `s(image, full_truth) > s(image, anchor) > s(image, half_truth)`.

Ties count as incorrect. Both metrics use all released rows with no additional filtering. Results include overall, entity, and relation counts, correct counts, percentages, and per-comparison scores. Overall scores weight rows equally; the three-source average weights configurations equally.
