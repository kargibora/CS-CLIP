#!/usr/bin/env python3
"""
Compute Triplet Embeddings for UMAP/t-SNE Analysis

This script loads models and computes embeddings for triplet geometry analysis:
- Image embeddings
- Unit (anchor/correct text) embeddings  
- Foil (incorrect text) embeddings

Outputs an NPZ file with:
- image_embs: (N, D) image embeddings
- unit_embs: (N, D) unit/anchor text embeddings
- foil_embs: (N, D) foil text embeddings
- metadata: dict with condition, sample_id, texts, etc.

Usage:
    python experiments/compute_triplet_embeddings.py \
        --json_path swap_pos_json/coco_val/ \
        --image_root . \
        --output_dir triplet_embeddings \
        --checkpoint_type openclip \
        --model_name ViT-B/32 \
        --num_samples 2000

    # For CS-CLIP
    python experiments/compute_triplet_embeddings.py \
        --json_path swap_pos_json/coco_val/ \
        --image_root . \
        --output_dir triplet_embeddings \
        --checkpoint_type external \
        --checkpoint_path /path/to/cs_clip.pt \
        --model_name ViT-B/32 \
        --output_name cs_clip
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
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
class TripletSample:
    """A triplet for embedding analysis: (Image, Unit, Foil).
    
    Now includes long_correct for full Half-Truth style analysis:
    - unit_text: Short correct (partial truth) - e.g., "dogs"
    - long_correct: Long correct (full truth) - e.g., "dogs on grass"  
    - foil_text: Long incorrect (half-truth) - e.g., "dogs on snow"
    """
    image_path: str
    sample_id: str
    condition: str          # e.g., 'component_easy', 'relation_swap'
    category: str           # 'entity' or 'relation'
    unit_text: str          # Short correct/anchor text (partial truth)
    foil_text: str          # Long incorrect/foil text (half-truth)
    long_correct: str       # Long correct text (full truth)
    tag: str                # More specific tag within condition
    original_caption: str = ""


# =============================================================================
# Sample Generator
# =============================================================================

def is_valid_caption(text: str) -> bool:
    """Check if a caption is valid (non-empty and meaningful)."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    words = stripped.split()
    meaningful_words = [w for w in words if len(w) > 2 and w.lower() not in {'and', 'the', 'a', 'an', 'on', 'in', 'at', 'to', 'of'}]
    return len(meaningful_words) > 0


class TripletSampleGenerator:
    """Generate triplet samples from structured JSON data."""
    
    def __init__(
        self,
        json_path: str,
        image_root: str,
        connector: str = " and ",
        max_samples: int = 2000,
        seed: int = 42,
    ):
        self.json_path = json_path
        self.image_root = image_root
        self.connector = connector
        self.max_samples = max_samples
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data
        logger.info(f"Loading data from {json_path}")
        self._load_data()
    
    def _load_data(self):
        """Load JSON data from file or folder."""
        json_path = Path(self.json_path)
        
        if json_path.is_file():
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        elif json_path.is_dir():
            self.data = []
            for json_file in sorted(json_path.glob("*.json")):
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        self.data.extend(file_data)
                    else:
                        self.data.append(file_data)
        else:
            raise ValueError(f"Invalid json_path: {json_path}")
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def generate_triplets(self) -> List[TripletSample]:
        """Generate triplet samples for all conditions (both entity and relation)."""
        entity_triplets = []
        relation_triplets = []
        used_images_entity = set()
        used_images_relation = set()
        
        # Calculate per-category limits
        max_per_category = self.max_samples // 2
        
        for sample in tqdm(self.data, desc="Generating triplets"):
            positive_components = sample.get('positive_components', [])
            negative_components = sample.get('negative_components', {})
            relations = sample.get('relations', [])
            
            # Get image path
            image_path = sample.get('image_path', '')
            if image_path and self.image_root and not image_path.startswith('/'):
                image_path = os.path.join(self.image_root, image_path)
            
            if not image_path or not os.path.exists(image_path):
                continue
            
            image_key = sample.get('sample_id', '')
            base_image_id = image_key.rsplit('_', 1)[0] if image_key else ''
            original_caption = sample.get('original_caption', '')
            
            # ===== ENTITY TRIPLETS (Component-based) =====
            if len(entity_triplets) < max_per_category:
                # Skip if already used this image for entity
                if not (base_image_id and base_image_id in used_images_entity):
                    added_entity = False
                    if len(positive_components) >= 2 and negative_components:
                        for i, comp_a in enumerate(positive_components):
                            if added_entity:
                                break
                            for j, comp_b in enumerate(positive_components):
                                if i == j or added_entity:
                                    continue
                                
                                comp_b_negatives = negative_components.get(comp_b, [])
                                if not comp_b_negatives:
                                    continue
                                
                                for neg_info in comp_b_negatives:
                                    if not isinstance(neg_info, dict):
                                        continue
                                    
                                    neg_text = neg_info.get('negative', '')
                                    change_type = neg_info.get('change_type', '')
                                    
                                    if not is_valid_caption(neg_text):
                                        continue
                                    
                                    short_correct = comp_a.lower().strip()
                                    comp_b_text = comp_b.lower().strip()
                                    if not is_valid_caption(short_correct) or not is_valid_caption(comp_b_text):
                                        continue
                                    
                                    # Unit = short correct caption (partial truth)
                                    # Long correct = short correct + correct component (full truth)
                                    # Foil = short correct + wrong component (half-truth)
                                    unit_text = short_correct
                                    long_correct = f"{short_correct}{self.connector}{comp_b_text}"
                                    foil_text = f"{short_correct}{self.connector}{neg_text.lower()}"
                                    
                                    if change_type == 'object_change':
                                        condition = 'component_easy'
                                        tag = '+Obj'
                                    elif change_type == 'attribute_change':
                                        condition = 'component_hard'
                                        tag = '+Attr'
                                    else:
                                        continue
                                    
                                    entity_triplets.append(TripletSample(
                                        image_path=image_path,
                                        sample_id=image_key,
                                        condition=condition,
                                        category='entity',
                                        unit_text=unit_text,
                                        foil_text=foil_text,
                                        long_correct=long_correct,
                                        tag=tag,
                                        original_caption=original_caption,
                                    ))
                                    added_entity = True
                                    if base_image_id:
                                        used_images_entity.add(base_image_id)
                                    break
            
            # ===== RELATION TRIPLETS =====
            if len(relation_triplets) < max_per_category:
                # Skip if already used this image for relation
                if not (base_image_id and base_image_id in used_images_relation):
                    added_relation = False
                    if relations:
                        for rel in relations:
                            if not isinstance(rel, dict) or added_relation:
                                continue
                            
                            subject = rel.get('subject', '').lower().strip()
                            relation_type = rel.get('relation_type', '').lower().strip()
                            obj = rel.get('object', '').lower().strip()
                            
                            if not all([subject, relation_type, obj]):
                                continue
                            
                            # Unit = partial truth (just subject)
                            unit_text = subject
                            # Long correct = full truth (complete relation)
                            long_correct = f"{subject} {relation_type} {obj}"
                            
                            # Relation swap
                            foil_swap = f"{obj} {relation_type} {subject}"
                            if is_valid_caption(foil_swap):
                                relation_triplets.append(TripletSample(
                                    image_path=image_path,
                                    sample_id=image_key,
                                    condition='relation_swap',
                                    category='relation',
                                    unit_text=unit_text,
                                    long_correct=long_correct,
                                    foil_text=foil_swap,
                                    tag='Swap',
                                    original_caption=original_caption,
                                ))
                                added_relation = True
                            
                            # Check negatives within relation
                            relation_negatives = rel.get('negatives', [])
                            for neg_info in relation_negatives:
                                if not isinstance(neg_info, dict):
                                    continue
                                
                                change_type = neg_info.get('change_type', '').lower()
                                neg_subject = neg_info.get('subject', '').lower().strip()
                                neg_rel_type = neg_info.get('relation_type', '').lower().strip()
                                neg_obj = neg_info.get('object', '').lower().strip()
                                
                                if not all([is_valid_caption(neg_subject), is_valid_caption(neg_rel_type), is_valid_caption(neg_obj)]):
                                    continue
                                
                                foil_caption = f"{neg_subject} {neg_rel_type} {neg_obj}"
                                
                                if change_type == 'antonym':
                                    relation_triplets.append(TripletSample(
                                        image_path=image_path,
                                        sample_id=image_key,
                                        condition='relation_antonym',
                                        category='relation',
                                        unit_text=unit_text,
                                        long_correct=long_correct,
                                        foil_text=foil_caption,
                                        tag='Ant',
                                        original_caption=original_caption,
                                    ))
                                    added_relation = True
                                elif change_type == 'object_change':
                                    relation_triplets.append(TripletSample(
                                        image_path=image_path,
                                        sample_id=image_key,
                                        condition='object_wrong',
                                        category='relation',
                                        unit_text=unit_text,
                                        long_correct=long_correct,
                                        foil_text=foil_caption,
                                        tag='Obj',
                                        original_caption=original_caption,
                                    ))
                                    added_relation = True
                                elif change_type == 'subject_change':
                                    relation_triplets.append(TripletSample(
                                        image_path=image_path,
                                        sample_id=image_key,
                                        condition='subject_wrong',
                                        category='relation',
                                        unit_text=unit_text,
                                        long_correct=long_correct,
                                        foil_text=foil_caption,
                                        tag='Subj',
                                        original_caption=original_caption,
                                    ))
                                    added_relation = True
                            
                            if added_relation:
                                if base_image_id:
                                    used_images_relation.add(base_image_id)
                                break
            
            # Early exit if both categories are full
            if len(entity_triplets) >= max_per_category and len(relation_triplets) >= max_per_category:
                break
        
        # Combine triplets
        triplets = entity_triplets + relation_triplets
        
        logger.info(f"Generated {len(triplets)} triplet samples:")
        logger.info(f"  - Entity triplets: {len(entity_triplets)}")
        logger.info(f"  - Relation triplets: {len(relation_triplets)}")
        return triplets


# =============================================================================
# Embedding Computation
# =============================================================================

class TripletEmbedder:
    """Compute embeddings for triplets using a CLIP model."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "openclip",
        force_openclip: bool = False,
        pretrained: str = "openai",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on {self.device}")
        
        from utils.checkpoint_loader import load_checkpoint_model
        
        if checkpoint_type == "openclip":
            effective_path = model_name
        else:
            if not checkpoint_path:
                raise ValueError(f"checkpoint_path required for checkpoint_type='{checkpoint_type}'")
            effective_path = checkpoint_path
        
        self.model, self.preprocess, self.tokenize = load_checkpoint_model(
            checkpoint_type=checkpoint_type,
            checkpoint_path=effective_path,
            device=self.device,
            base_model=model_name,
            force_openclip=force_openclip,
            pretrained=pretrained,
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to normalized embedding."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()[0]
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to normalized embedding."""
        try:
            text_input = self.tokenize([text]).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_input = self.tokenize([text], truncate=True).to(self.device)
        features = self.model.encode_text(text_input)
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()[0]
    
    def compute_triplet_embeddings(
        self,
        triplets: List[TripletSample],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Compute embeddings for all triplets.
        
        Returns:
            Dict with keys:
            - image_embs: (N, D) numpy array
            - unit_embs: (N, D) numpy array (short correct / partial truth)
            - long_correct_embs: (N, D) numpy array (full correct truth)
            - foil_embs: (N, D) numpy array (half-truth / long incorrect)
            - sample_ids: list of sample IDs
            - conditions: list of condition names
            - categories: list of 'entity' or 'relation'
            - tags: list of tags
            - unit_texts: list of unit texts (short correct)
            - long_correct_texts: list of long correct texts
            - foil_texts: list of foil texts (long incorrect)
            - similarities: dict with d_plus, d_minus, gamma arrays
        """
        image_embs = []
        unit_embs = []
        long_correct_embs = []
        foil_embs = []
        sample_ids = []
        conditions = []
        categories = []
        tags = []
        unit_texts = []
        long_correct_texts = []
        foil_texts = []
        
        for triplet in tqdm(triplets, desc="Computing embeddings"):
            try:
                image = Image.open(triplet.image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {triplet.image_path}: {e}")
                continue
            
            # Compute embeddings
            img_emb = self.encode_image(image)
            unit_emb = self.encode_text(triplet.unit_text)
            long_correct_emb = self.encode_text(triplet.long_correct)
            foil_emb = self.encode_text(triplet.foil_text)
            
            image_embs.append(img_emb)
            unit_embs.append(unit_emb)
            long_correct_embs.append(long_correct_emb)
            foil_embs.append(foil_emb)
            sample_ids.append(triplet.sample_id)
            conditions.append(triplet.condition)
            categories.append(triplet.category)
            tags.append(triplet.tag)
            unit_texts.append(triplet.unit_text)
            long_correct_texts.append(triplet.long_correct)
            foil_texts.append(triplet.foil_text)
        
        # Convert to numpy arrays
        image_embs = np.array(image_embs)
        unit_embs = np.array(unit_embs)
        long_correct_embs = np.array(long_correct_embs)
        foil_embs = np.array(foil_embs)
        
        # Compute triplet geometry metrics
        # For Half-Truth style analysis:
        # short_correct (unit) = partial truth
        # long_correct = full truth
        # long_incorrect (foil) = half-truth
        
        # d+ = 1 - s(I, U)  (distance to short correct)
        # d- = 1 - s(I, F)  (distance to foil/long incorrect)
        # gamma = d- - d+   (separation: positive means short correct closer to image)
        
        sim_image_unit = np.sum(image_embs * unit_embs, axis=1)  # s(I, short_correct)
        sim_image_long_correct = np.sum(image_embs * long_correct_embs, axis=1)  # s(I, long_correct)
        sim_image_foil = np.sum(image_embs * foil_embs, axis=1)  # s(I, long_incorrect)
        
        sim_unit_foil = np.sum(unit_embs * foil_embs, axis=1)    # s(short_correct, long_incorrect)
        sim_unit_long_correct = np.sum(unit_embs * long_correct_embs, axis=1)  # s(short_correct, long_correct)
        sim_long_correct_foil = np.sum(long_correct_embs * foil_embs, axis=1)  # s(long_correct, long_incorrect)
        
        d_plus = 1 - sim_image_unit
        d_minus = 1 - sim_image_foil
        gamma = d_minus - d_plus  # separation
        
        return {
            'image_embs': image_embs,
            'unit_embs': unit_embs,
            'long_correct_embs': long_correct_embs,
            'foil_embs': foil_embs,
            'sample_ids': sample_ids,
            'conditions': conditions,
            'categories': categories,
            'tags': tags,
            'unit_texts': unit_texts,
            'long_correct_texts': long_correct_texts,
            'foil_texts': foil_texts,
            # Image-text similarities
            'sim_image_unit': sim_image_unit,
            'sim_image_long_correct': sim_image_long_correct,
            'sim_image_foil': sim_image_foil,
            # Text-text similarities
            'sim_unit_foil': sim_unit_foil,
            'sim_unit_long_correct': sim_unit_long_correct,
            'sim_long_correct_foil': sim_long_correct_foil,
            # Triplet geometry
            'd_plus': d_plus,
            'd_minus': d_minus,
            'gamma': gamma,
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute triplet embeddings for UMAP analysis")
    
    # Data arguments
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to JSON file or folder with structured data")
    parser.add_argument("--image_root", type=str, default=".",
                        help="Root directory for images")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Maximum number of triplet samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="ViT-B/32",
                        help="CLIP model architecture")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                        choices=["openclip", "external", "huggingface", "dac", "clove"],
                        help="Type of checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--force_openclip", action="store_true",
                        help="Force OpenCLIP instead of OpenAI CLIP")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="triplet_embeddings",
                        help="Output directory")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename prefix (default: derived from model)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate triplets
    generator = TripletSampleGenerator(
        json_path=args.json_path,
        image_root=args.image_root,
        max_samples=args.num_samples,
        seed=args.seed,
    )
    triplets = generator.generate_triplets()
    
    if len(triplets) == 0:
        logger.error("No triplets generated!")
        return
    
    # Compute embeddings
    embedder = TripletEmbedder(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        checkpoint_type=args.checkpoint_type,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
    )
    
    results = embedder.compute_triplet_embeddings(triplets)
    
    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    elif args.checkpoint_type == "openclip":
        output_name = args.model_name.replace("/", "_").replace("-", "_").lower()
    else:
        output_name = Path(args.checkpoint_path).stem if args.checkpoint_path else "model"
    
    output_path = os.path.join(args.output_dir, f"{output_name}_triplet_embeddings.npz")
    
    # Save results
    np.savez_compressed(
        output_path,
        # Embeddings
        image_embs=results['image_embs'],
        unit_embs=results['unit_embs'],
        long_correct_embs=results['long_correct_embs'],
        foil_embs=results['foil_embs'],
        # Metadata
        sample_ids=np.array(results['sample_ids'], dtype=object),
        conditions=np.array(results['conditions'], dtype=object),
        categories=np.array(results['categories'], dtype=object),
        tags=np.array(results['tags'], dtype=object),
        # Texts
        unit_texts=np.array(results['unit_texts'], dtype=object),
        long_correct_texts=np.array(results['long_correct_texts'], dtype=object),
        foil_texts=np.array(results['foil_texts'], dtype=object),
        # Image-text similarities
        sim_image_unit=results['sim_image_unit'],
        sim_image_long_correct=results['sim_image_long_correct'],
        sim_image_foil=results['sim_image_foil'],
        # Text-text similarities
        sim_unit_foil=results['sim_unit_foil'],
        sim_unit_long_correct=results['sim_unit_long_correct'],
        sim_long_correct_foil=results['sim_long_correct_foil'],
        # Triplet geometry
        d_plus=results['d_plus'],
        d_minus=results['d_minus'],
        gamma=results['gamma'],
    )
    
    logger.info(f"Saved embeddings to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TRIPLET EMBEDDING SUMMARY")
    print("="*70)
    print(f"Total triplets: {len(results['sample_ids'])}")
    print(f"Embedding dimension: {results['image_embs'].shape[1]}")
    
    # Stats by category
    categories = np.array(results['categories'])
    for cat in ['entity', 'relation']:
        mask = categories == cat
        if mask.sum() > 0:
            gamma_cat = results['gamma'][mask]
            print(f"\n{cat.upper()}:")
            print(f"  Count: {mask.sum()}")
            print(f"  Gamma (separation): mean={gamma_cat.mean():.4f}, median={np.median(gamma_cat):.4f}")
            print(f"  Pr[gamma > 0] (unit closer): {(gamma_cat > 0).mean() * 100:.1f}%")
    
    # Stats by tag
    tags = np.array(results['tags'])
    print("\nBy Tag:")
    for tag in np.unique(tags):
        mask = tags == tag
        gamma_tag = results['gamma'][mask]
        print(f"  {tag}: n={mask.sum()}, Pr[γ>0]={(gamma_tag > 0).mean() * 100:.1f}%, mean γ={gamma_tag.mean():.4f}")


if __name__ == "__main__":
    main()
