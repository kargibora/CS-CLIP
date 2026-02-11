#!/usr/bin/env python3
"""
Compute and Save Embeddings for Visualization

Computes image and text embeddings for samples and saves them to disk.
Run this once per model, then load the embeddings in the visualization notebook.

Output structure:
    output_dir/
        embeddings.npz          # All embeddings (image + text)
        metadata.json           # Sample info, texts, categories
        images/                 # Thumbnail images (optional)
            sample_000.jpg
            ...

Usage:
    # Baseline CLIP
    python experiments/compute_embeddings.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --output_dir embeddings/clip_baseline \
        --num_samples 100 \
        --model_name "CLIP Baseline"

    # Fine-tuned model
    python experiments/compute_embeddings.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --checkpoint_path /path/to/finetuned.pt \
        --checkpoint_type external \
        --output_dir embeddings/finetuned \
        --model_name "Our Model"
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Sample:
    """Sample with image and text variants."""
    image_path: str
    sample_id: str
    original_caption: str
    components: List[str]
    component_negatives: Dict[str, List[Dict[str, str]]]
    relations: List[Dict[str, str]]
    binding_negatives: List[Dict[str, str]]


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    checkpoint_type: str = "openclip",
    force_openclip: bool = False,
    pretrained: str = "openai",
):
    """Load CLIP model."""
    from utils.checkpoint_loader import load_checkpoint_model
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    effective_path = model_name if checkpoint_type == "openclip" else checkpoint_path
    
    model, preprocess, tokenize = load_checkpoint_model(
        checkpoint_type=checkpoint_type,
        checkpoint_path=effective_path,
        device=device,
        base_model=model_name,
        force_openclip=force_openclip,
        pretrained=pretrained,
    )
    model.eval()
    
    return model, preprocess, tokenize, device


# =============================================================================
# Data Loading
# =============================================================================

def load_samples(
    json_folder: str,
    image_root: str,
    max_samples: int = 100,
    seed: int = 42,
) -> List[Sample]:
    """Load samples from JSON files."""
    random.seed(seed)
    np.random.seed(seed)
    
    json_folder = Path(json_folder)
    samples = []
    
    json_files = sorted(json_folder.glob("*.json"))
    if not json_files:
        if json_folder.is_file():
            json_files = [json_folder]
        else:
            raise ValueError(f"No JSON files found in {json_folder}")
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    random.shuffle(all_data)
    
    for item in all_data:
        if len(samples) >= max_samples:
            break
        
        image_path = item.get('image_path', '')
        if image_path and image_root and not image_path.startswith('/'):
            image_path = os.path.join(image_root, image_path)
        
        if not image_path or not os.path.exists(image_path):
            continue
        
        components = item.get('positive_components', [])
        if not components:
            continue
        
        samples.append(Sample(
            image_path=image_path,
            sample_id=item.get('sample_id', f"sample_{len(samples)}"),
            original_caption=item.get('caption', item.get('original_caption', '')),
            components=components,
            component_negatives=item.get('negative_components', {}),
            relations=item.get('relations', []),
            binding_negatives=item.get('binding_negatives', []),
        ))
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


# =============================================================================
# Embedding Computation
# =============================================================================

@torch.no_grad()
def encode_image(model, preprocess, image_path: str, device) -> np.ndarray:
    """Encode image to normalized embedding."""
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    features = model.encode_image(image_input)
    features = F.normalize(features, dim=-1)
    return features.cpu().numpy()[0]


@torch.no_grad()
def encode_text(model, tokenize, text: str, device) -> np.ndarray:
    """Encode text to normalized embedding."""
    try:
        text_input = tokenize([text]).to(device)
    except (RuntimeError, ValueError, TypeError):
        text_input = tokenize([text], truncate=True).to(device)
    features = model.encode_text(text_input)
    features = F.normalize(features, dim=-1)
    return features.cpu().numpy()[0]


def compute_sample_embeddings(
    model, preprocess, tokenize, device,
    sample: Sample,
    max_components: int = 3,
    max_negatives: int = 2,
) -> Optional[Dict[str, Any]]:
    """Compute all embeddings for a single sample."""
    try:
        image_emb = encode_image(model, preprocess, sample.image_path, device)
    except Exception as e:
        logger.warning(f"Failed to load image {sample.image_path}: {e}")
        return None
    
    result = {
        'sample_id': sample.sample_id,
        'image_path': sample.image_path,
        'original_caption': sample.original_caption,
        'image_emb': image_emb,
        'texts': [],
        'text_embs': [],
        'categories': [],
        'is_positive': [],
        'change_types': [],  # Track what type of negative it is
    }
    
    # Original caption
    result['texts'].append(sample.original_caption)
    result['text_embs'].append(encode_text(model, tokenize, sample.original_caption, device))
    result['categories'].append('caption')
    result['is_positive'].append(True)
    result['change_types'].append('original')
    
    # Components (positive)
    for comp in sample.components[:max_components]:
        result['texts'].append(comp)
        result['text_embs'].append(encode_text(model, tokenize, comp, device))
        result['categories'].append('component_pos')
        result['is_positive'].append(True)
        result['change_types'].append('original')
        
        # Component negatives
        negs = sample.component_negatives.get(comp, [])
        for neg in negs[:max_negatives]:
            neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
            change_type = neg.get('change_type', 'unknown') if isinstance(neg, dict) else 'unknown'
            
            if neg_text:
                result['texts'].append(neg_text)
                result['text_embs'].append(encode_text(model, tokenize, neg_text, device))
                result['categories'].append('component_neg')
                result['is_positive'].append(False)
                result['change_types'].append(change_type)
    
    # Relations
    for rel in sample.relations[:2]:
        subject = rel.get('subject', '')
        relation_type = rel.get('relation_type', '')
        obj = rel.get('object', '')
        
        if subject and relation_type and obj:
            # Correct relation
            rel_text = f"{subject} {relation_type} {obj}"
            result['texts'].append(rel_text)
            result['text_embs'].append(encode_text(model, tokenize, rel_text, device))
            result['categories'].append('relation_correct')
            result['is_positive'].append(True)
            result['change_types'].append('original')
            
            # Swapped relation (subject <-> object)
            swapped = f"{obj} {relation_type} {subject}"
            result['texts'].append(swapped)
            result['text_embs'].append(encode_text(model, tokenize, swapped, device))
            result['categories'].append('relation_swapped')
            result['is_positive'].append(False)
            result['change_types'].append('swap')
            
            # Relation negatives (if available)
            for rel_neg in rel.get('negatives', [])[:1]:
                neg_rel = rel_neg.get('relation_type', '')
                neg_change = rel_neg.get('change_type', 'relation_change')
                if neg_rel:
                    neg_text = f"{subject} {neg_rel} {obj}"
                    result['texts'].append(neg_text)
                    result['text_embs'].append(encode_text(model, tokenize, neg_text, device))
                    result['categories'].append('relation_neg')
                    result['is_positive'].append(False)
                    result['change_types'].append(neg_change)
    
    # Binding pairs
    for binding in sample.binding_negatives[:2]:
        comp1 = binding.get('component_1', '')
        comp2 = binding.get('component_2', '')
        bind_neg1 = binding.get('binding_neg_1', '')
        bind_neg2 = binding.get('binding_neg_2', '')
        
        if comp1 and bind_neg1:
            # Original component
            result['texts'].append(comp1)
            result['text_embs'].append(encode_text(model, tokenize, comp1, device))
            result['categories'].append('binding_pos')
            result['is_positive'].append(True)
            result['change_types'].append('original')
            
            # Binding negative (swapped attributes)
            result['texts'].append(bind_neg1)
            result['text_embs'].append(encode_text(model, tokenize, bind_neg1, device))
            result['categories'].append('binding_neg')
            result['is_positive'].append(False)
            result['change_types'].append('attribute_swap')
        
        if comp2 and bind_neg2:
            result['texts'].append(comp2)
            result['text_embs'].append(encode_text(model, tokenize, comp2, device))
            result['categories'].append('binding_pos')
            result['is_positive'].append(True)
            result['change_types'].append('original')
            
            result['texts'].append(bind_neg2)
            result['text_embs'].append(encode_text(model, tokenize, bind_neg2, device))
            result['categories'].append('binding_neg')
            result['is_positive'].append(False)
            result['change_types'].append('attribute_swap')
    
    result['text_embs'] = np.array(result['text_embs'])
    
    return result


def compute_all_embeddings(
    model, preprocess, tokenize, device,
    samples: List[Sample],
    save_thumbnails: bool = True,
    thumbnail_dir: Optional[Path] = None,
    thumbnail_size: int = 224,
) -> Dict[str, Any]:
    """Compute embeddings for all samples."""
    
    # Per-sample data
    sample_ids = []
    image_paths = []
    original_captions = []
    
    # Embeddings
    image_embeddings = []
    text_embeddings = []
    
    # Text metadata (flattened across all samples)
    all_texts = []
    all_categories = []
    all_is_positive = []
    all_change_types = []
    all_text_sample_ids = []  # Which sample each text belongs to
    
    # Text index mapping (which texts belong to which sample)
    text_indices_per_sample = []
    
    current_text_idx = 0
    
    for i, sample in enumerate(tqdm(samples, desc="Computing embeddings")):
        data = compute_sample_embeddings(model, preprocess, tokenize, device, sample)
        if data is None:
            continue
        
        sample_ids.append(data['sample_id'])
        image_paths.append(data['image_path'])
        original_captions.append(data['original_caption'])
        image_embeddings.append(data['image_emb'])
        
        # Track text indices for this sample
        n_texts = len(data['texts'])
        text_indices_per_sample.append(list(range(current_text_idx, current_text_idx + n_texts)))
        current_text_idx += n_texts
        
        # Add text data
        for j in range(n_texts):
            text_embeddings.append(data['text_embs'][j])
            all_texts.append(data['texts'][j])
            all_categories.append(data['categories'][j])
            all_is_positive.append(data['is_positive'][j])
            all_change_types.append(data['change_types'][j])
            all_text_sample_ids.append(data['sample_id'])
        
        # Save thumbnail
        if save_thumbnails and thumbnail_dir:
            try:
                img = Image.open(data['image_path']).convert('RGB')
                img.thumbnail((thumbnail_size, thumbnail_size))
                img.save(thumbnail_dir / f"sample_{i:04d}.jpg", quality=85)
            except Exception as e:
                logger.warning(f"Failed to save thumbnail: {e}")
    
    return {
        # Sample-level data
        'sample_ids': sample_ids,
        'image_paths': image_paths,
        'original_captions': original_captions,
        'image_embeddings': np.array(image_embeddings),
        'text_indices_per_sample': text_indices_per_sample,
        
        # Text-level data (flattened)
        'text_embeddings': np.array(text_embeddings),
        'texts': all_texts,
        'categories': all_categories,
        'is_positive': all_is_positive,
        'change_types': all_change_types,
        'text_sample_ids': all_text_sample_ids,
    }


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_embeddings(data: Dict[str, Any], output_dir: Path, model_name: str):
    """Save embeddings and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as npz
    np.savez_compressed(
        output_dir / "embeddings.npz",
        image_embeddings=data['image_embeddings'],
        text_embeddings=data['text_embeddings'],
    )
    
    # Save metadata as JSON
    metadata = {
        'model_name': model_name,
        'n_samples': len(data['sample_ids']),
        'n_texts': len(data['texts']),
        'embedding_dim': data['image_embeddings'].shape[1] if len(data['image_embeddings']) > 0 else 0,
        
        # Sample-level
        'sample_ids': data['sample_ids'],
        'image_paths': data['image_paths'],
        'original_captions': data['original_captions'],
        'text_indices_per_sample': data['text_indices_per_sample'],
        
        # Text-level
        'texts': data['texts'],
        'categories': data['categories'],
        'is_positive': data['is_positive'],
        'change_types': data['change_types'],
        'text_sample_ids': data['text_sample_ids'],
        
        # Category summary
        'category_counts': {
            cat: data['categories'].count(cat)
            for cat in set(data['categories'])
        },
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved embeddings to {output_dir}")
    logger.info(f"  - {len(data['sample_ids'])} samples")
    logger.info(f"  - {len(data['texts'])} text embeddings")
    logger.info(f"  - Embedding dimension: {metadata['embedding_dim']}")


def load_embeddings(embedding_dir: Path) -> Dict[str, Any]:
    """Load saved embeddings and metadata."""
    embedding_dir = Path(embedding_dir)
    
    # Load embeddings
    emb_data = np.load(embedding_dir / "embeddings.npz")
    
    # Load metadata
    with open(embedding_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'image_embeddings': emb_data['image_embeddings'],
        'text_embeddings': emb_data['text_embeddings'],
        **metadata,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute and Save Embeddings")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="ViT-B/32",
                       help="Base CLIP model architecture")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "huggingface", "tripletclip", "external", "dac", "clove"])
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to fine-tuned checkpoint")
    parser.add_argument("--force_openclip", action="store_true")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Human-readable model name for metadata")
    
    # Data arguments
    parser.add_argument("--json_folder", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_thumbnails", action="store_true", default=True)
    parser.add_argument("--thumbnail_size", type=int, default=224)
    
    args = parser.parse_args()
    
    # Set model name
    if args.model_name is None:
        if args.checkpoint_path:
            args.model_name = Path(args.checkpoint_path).stem
        else:
            args.model_name = f"CLIP ({args.base_model})"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create thumbnail directory
    thumbnail_dir = None
    if args.save_thumbnails:
        thumbnail_dir = output_dir / "thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model, preprocess, tokenize, device = load_model(
        model_name=args.base_model,
        checkpoint_path=args.checkpoint_path,
        checkpoint_type=args.checkpoint_type,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
    )
    
    # Load samples
    samples = load_samples(
        args.json_folder,
        args.image_root,
        max_samples=args.num_samples,
        seed=args.seed,
    )
    
    if not samples:
        logger.error("No samples loaded!")
        return
    
    # Compute embeddings
    logger.info("Computing embeddings...")
    data = compute_all_embeddings(
        model, preprocess, tokenize, device,
        samples,
        save_thumbnails=args.save_thumbnails,
        thumbnail_dir=thumbnail_dir,
        thumbnail_size=args.thumbnail_size,
    )
    
    # Save
    save_embeddings(data, output_dir, args.model_name)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EMBEDDING COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")
    print(f"Samples: {len(data['sample_ids'])}")
    print(f"Text embeddings: {len(data['texts'])}")
    print(f"\nCategory distribution:")
    for cat in sorted(set(data['categories'])):
        count = data['categories'].count(cat)
        print(f"  {cat}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
