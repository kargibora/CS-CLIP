"""Evaluate the released Half-Truth comparisons with CLIP or a CS-CLIP checkpoint."""
import argparse
import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.clip_wrapper import load_clip_model
from utils.checkpoint_loader import clean_state_dict, get_tokenizer


def load_records(path, subset):
    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download('kbora/Half-Truths', f'data/{subset}.parquet', repo_type='dataset'))
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines()]
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def summarize(rows):
    result = {}
    for kind in ('all', 'entity', 'relation'):
        selected = [r for r in rows if kind == 'all' or r['triplet_type'] == kind]
        n = len(selected)
        ht = sum(r['s_anchor'] > r['s_half_truth'] for r in selected)
        cor = sum(r['s_full_truth'] > r['s_anchor'] > r['s_half_truth'] for r in selected)
        result[kind] = {'n': n, 'ht_correct': ht, 'ht_accuracy': 100 * ht / n if n else None,
                        'cor_correct': cor, 'cor_accuracy': 100 * cor / n if n else None}
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=['coco_qwen', 'coco_mistral', 'cc3m_qwen'], default='coco_qwen')
    parser.add_argument('--data', type=Path, help='Local release Parquet or JSONL; otherwise download from Hugging Face')
    parser.add_argument('--image-root', type=Path, default=Path('.'), help='Image root for local JSONL inputs')
    parser.add_argument('--checkpoint', type=Path, help='CS-CLIP checkpoint; omit for pretrained CLIP')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path('half_truth_results.json'))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    device = torch.device(args.device)
    fine_tuned = args.checkpoint is not None
    model, preprocess = load_clip_model('ViT-B/32', device, force_openclip=fine_tuned)
    if fine_tuned:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        state = {k: v for k, v in clean_state_dict(state).items() if not k.startswith('head.')}
        model.load_state_dict(state, strict=True)
    model.eval()
    tokenize = get_tokenizer('ViT-B/32', 'openclip' if fine_tuned else 'openai')
    records = load_records(args.data, args.subset)
    if not records:
        raise ValueError('Evaluation set is empty')
    scores = []
    for start in tqdm(range(0, len(records), args.batch_size)):
        batch = records[start:start + args.batch_size]
        images = []
        for row in batch:
            source = io.BytesIO(row['image']['bytes']) if isinstance(row['image'], dict) else args.image_root / row['image']
            with Image.open(source) as im:
                images.append(preprocess(im.convert('RGB')))
        image_features = F.normalize(model.encode_image(torch.stack(images).to(device)), dim=-1)
        batch_scores = {}
        for caption in ('anchor', 'half_truth', 'full_truth'):
            tokens = tokenize([r[caption] for r in batch]).to(device)
            text_features = F.normalize(model.encode_text(tokens), dim=-1)
            batch_scores[caption] = torch.stack([
                torch.dot(image, text) for image, text in zip(image_features, text_features)
            ]).float().cpu().tolist()
        for index, row in enumerate(batch):
            scores.append({'id': row['id'], 'triplet_type': row['triplet_type'],
                           **{f's_{key}': values[index] for key, values in batch_scores.items()}})
    output = {'subset': args.subset, 'model': str(args.checkpoint) if fine_tuned else 'CLIP',
              'summary': summarize(scores), 'scores': scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['summary'], indent=2))


if __name__ == '__main__':
    main()
