#!/usr/bin/env python3
"""
Embedding Visualization for Components, Relations, and Half-Truths

Visualizes how CLIP models embed:
1. Components vs their negatives (attribute/object changes)
2. Relations vs swapped/wrong relations
3. Full captions vs half-truth foils
4. Binding pairs (attribute-noun swaps)

Creates:
- t-SNE/UMAP plots of text embeddings
- Similarity heatmaps between image and text variants
- Embedding trajectory plots (component -> relation -> full caption)

Usage:
    python experiments/embedding_visualization.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --output_dir embedding_viz_results \
        --num_samples 100
        
    # With fine-tuned model
    python experiments/embedding_visualization.py \
        --checkpoint_path /path/to/checkpoint.pt \
        --checkpoint_type external \
        --output_dir embedding_viz_finetuned
"""

import os
import sys
import json
import random
import argparse
import logging
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

from utils.checkpoint_loader import load_checkpoint_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EmbeddingSample:
    """A sample with multiple text variants for embedding analysis."""
    image_path: str
    sample_id: str
    original_caption: str
    
    # Component-level texts
    components: List[str]
    component_negatives: Dict[str, List[Dict[str, str]]]  # comp -> [{negative, change_type}]
    
    # Relation-level texts
    relations: List[Dict[str, str]]  # [{subject, relation_type, object}]
    relation_negatives: List[Dict[str, Any]]  # Negative relations
    
    # Binding pairs (if available)
    binding_negatives: List[Dict[str, str]]  # [{component_1, component_2, binding_neg_1, binding_neg_2}]


# =============================================================================
# Embedding Extractor
# =============================================================================

class EmbeddingExtractor:
    """Extracts embeddings from CLIP models for visualization."""
    
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
        
        # Load model
        effective_path = model_name if checkpoint_type == "openclip" else checkpoint_path
        
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
        """Encode a single image."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()[0]
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        if not texts:
            return np.array([])
        
        try:
            text_input = self.tokenize(texts).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_input = self.tokenize(texts, truncate=True).to(self.device)
        
        features = self.model.encode_text(text_input)
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode_texts([text])[0]
    
    def compute_similarity_matrix(
        self,
        image: Image.Image,
        texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute similarity matrix between image and texts, and text-text similarities.
        
        Returns:
            image_emb: Image embedding (D,)
            text_embs: Text embeddings (N, D)
            sim_matrix: Similarity matrix - first row is image-text, rest is text-text (N+1, N)
        """
        image_emb = self.encode_image(image)
        text_embs = self.encode_texts(texts)
        
        # Image-text similarities
        img_text_sim = image_emb @ text_embs.T  # (N,)
        
        # Text-text similarities
        text_text_sim = text_embs @ text_embs.T  # (N, N)
        
        # Combine: first row is image-text, rest is text-text
        sim_matrix = np.vstack([img_text_sim.reshape(1, -1), text_text_sim])
        
        return image_emb, text_embs, sim_matrix


# =============================================================================
# Sample Loader
# =============================================================================

def load_samples(
    json_folder: str,
    image_root: str,
    max_samples: int = 100,
    seed: int = 42,
) -> List[EmbeddingSample]:
    """Load samples from JSON files."""
    random.seed(seed)
    np.random.seed(seed)
    
    json_folder = Path(json_folder)
    samples = []
    
    # Find JSON files
    json_files = sorted(json_folder.glob("*.json"))
    if not json_files:
        # Maybe it's a single file
        if json_folder.is_file():
            json_files = [json_folder]
        else:
            raise ValueError(f"No JSON files found in {json_folder}")
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Load all samples
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    logger.info(f"Loaded {len(all_data)} total samples")
    
    # Shuffle and subsample
    random.shuffle(all_data)
    
    for item in all_data:
        if len(samples) >= max_samples:
            break
        
        # Get image path
        image_path = item.get('image_path', '')
        if image_path and image_root and not image_path.startswith('/'):
            image_path = os.path.join(image_root, image_path)
        
        if not image_path or not os.path.exists(image_path):
            continue
        
        # Extract components and negatives
        components = item.get('positive_components', [])
        component_negatives = item.get('negative_components', {})
        
        if not components:
            continue
        
        # Extract relations
        relations = item.get('relations', [])
        relation_negatives = item.get('negative_relations', [])
        
        # For COCO format, negatives might be inside relations
        if not relation_negatives:
            for rel in relations:
                negs = rel.get('negatives', [])
                for neg in negs:
                    relation_negatives.append({
                        'subject': rel.get('subject', ''),
                        'relation_type': neg.get('relation_type', rel.get('relation_type', '')),
                        'object': rel.get('object', ''),
                        'change_type': neg.get('change_type', ''),
                        'original_relation': rel.get('relation_type', ''),
                    })
        
        # Extract binding negatives
        binding_negatives = item.get('binding_negatives', [])
        
        samples.append(EmbeddingSample(
            image_path=image_path,
            sample_id=item.get('sample_id', f"sample_{len(samples)}"),
            original_caption=item.get('caption', item.get('original_caption', '')),
            components=components,
            component_negatives=component_negatives,
            relations=relations,
            relation_negatives=relation_negatives,
            binding_negatives=binding_negatives,
        ))
    
    logger.info(f"Prepared {len(samples)} valid samples")
    return samples


# =============================================================================
# Visualization Functions
# =============================================================================

def create_component_embedding_data(
    extractor: EmbeddingExtractor,
    samples: List[EmbeddingSample],
    max_per_sample: int = 3,
) -> Dict[str, Any]:
    """Extract embeddings for components and their negatives.
    
    Returns dict with:
        - embeddings: (N, D) array of all embeddings
        - labels: List of labels for each embedding
        - categories: List of category for each embedding (positive/negative_attr/negative_obj)
        - sample_ids: List of sample IDs
        - image_embeddings: Dict[sample_id -> embedding]
    """
    embeddings = []
    labels = []
    categories = []
    sample_ids = []
    image_embeddings = {}
    
    for sample in tqdm(samples, desc="Extracting component embeddings"):
        try:
            image = Image.open(sample.image_path).convert('RGB')
            img_emb = extractor.encode_image(image)
            image_embeddings[sample.sample_id] = img_emb
        except Exception as e:
            logger.warning(f"Failed to load image {sample.image_path}: {e}")
            continue
        
        for comp in sample.components[:max_per_sample]:
            # Positive component
            comp_emb = extractor.encode_text(comp)
            embeddings.append(comp_emb)
            labels.append(comp)
            categories.append("positive")
            sample_ids.append(sample.sample_id)
            
            # Negative components
            negs = sample.component_negatives.get(comp, [])
            for neg in negs[:2]:  # Max 2 negatives per component
                neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
                change_type = neg.get('change_type', 'unknown') if isinstance(neg, dict) else 'unknown'
                
                if not neg_text:
                    continue
                
                neg_emb = extractor.encode_text(neg_text)
                embeddings.append(neg_emb)
                labels.append(neg_text)
                categories.append(f"negative_{change_type}")
                sample_ids.append(sample.sample_id)
    
    return {
        'embeddings': np.array(embeddings),
        'labels': labels,
        'categories': categories,
        'sample_ids': sample_ids,
        'image_embeddings': image_embeddings,
    }


def create_relation_embedding_data(
    extractor: EmbeddingExtractor,
    samples: List[EmbeddingSample],
) -> Dict[str, Any]:
    """Extract embeddings for relations and their variants."""
    embeddings = []
    labels = []
    categories = []
    sample_ids = []
    
    for sample in tqdm(samples, desc="Extracting relation embeddings"):
        for rel in sample.relations[:2]:  # Max 2 relations per sample
            subject = rel.get('subject', '')
            relation_type = rel.get('relation_type', '')
            obj = rel.get('object', '')
            
            if not subject or not relation_type or not obj:
                continue
            
            # Original relation
            rel_text = f"{subject} {relation_type} {obj}"
            rel_emb = extractor.encode_text(rel_text)
            embeddings.append(rel_emb)
            labels.append(rel_text)
            categories.append("original")
            sample_ids.append(sample.sample_id)
            
            # Subject only (partial)
            subj_emb = extractor.encode_text(subject)
            embeddings.append(subj_emb)
            labels.append(subject)
            categories.append("partial_subject")
            sample_ids.append(sample.sample_id)
            
            # Swapped relation
            swapped_text = f"{obj} {relation_type} {subject}"
            swapped_emb = extractor.encode_text(swapped_text)
            embeddings.append(swapped_emb)
            labels.append(swapped_text)
            categories.append("swapped")
            sample_ids.append(sample.sample_id)
        
        # Add negative relations
        for neg_rel in sample.relation_negatives[:2]:
            subject = neg_rel.get('subject', '')
            relation_type = neg_rel.get('relation_type', '')
            obj = neg_rel.get('object', '')
            change_type = neg_rel.get('change_type', 'unknown')
            
            if not subject or not relation_type or not obj:
                continue
            
            neg_text = f"{subject} {relation_type} {obj}"
            neg_emb = extractor.encode_text(neg_text)
            embeddings.append(neg_emb)
            labels.append(neg_text)
            categories.append(f"negative_{change_type}")
            sample_ids.append(sample.sample_id)
    
    return {
        'embeddings': np.array(embeddings),
        'labels': labels,
        'categories': categories,
        'sample_ids': sample_ids,
    }


def create_binding_embedding_data(
    extractor: EmbeddingExtractor,
    samples: List[EmbeddingSample],
) -> Dict[str, Any]:
    """Extract embeddings for binding pairs (attribute-noun swaps)."""
    embeddings = []
    labels = []
    categories = []
    sample_ids = []
    pair_ids = []
    
    pair_counter = 0
    
    for sample in tqdm(samples, desc="Extracting binding embeddings"):
        for binding in sample.binding_negatives[:3]:  # Max 3 binding pairs per sample
            comp1 = binding.get('component_1', '')
            comp2 = binding.get('component_2', '')
            bind_neg1 = binding.get('binding_neg_1', '')
            bind_neg2 = binding.get('binding_neg_2', '')
            
            if not comp1 or not comp2 or not bind_neg1 or not bind_neg2:
                continue
            
            # Original components
            emb1 = extractor.encode_text(comp1)
            embeddings.append(emb1)
            labels.append(comp1)
            categories.append("original_1")
            sample_ids.append(sample.sample_id)
            pair_ids.append(pair_counter)
            
            emb2 = extractor.encode_text(comp2)
            embeddings.append(emb2)
            labels.append(comp2)
            categories.append("original_2")
            sample_ids.append(sample.sample_id)
            pair_ids.append(pair_counter)
            
            # Binding negatives (swapped nouns)
            neg_emb1 = extractor.encode_text(bind_neg1)
            embeddings.append(neg_emb1)
            labels.append(bind_neg1)
            categories.append("binding_neg_1")
            sample_ids.append(sample.sample_id)
            pair_ids.append(pair_counter)
            
            neg_emb2 = extractor.encode_text(bind_neg2)
            embeddings.append(neg_emb2)
            labels.append(bind_neg2)
            categories.append("binding_neg_2")
            sample_ids.append(sample.sample_id)
            pair_ids.append(pair_counter)
            
            pair_counter += 1
    
    return {
        'embeddings': np.array(embeddings) if embeddings else np.array([]).reshape(0, 512),
        'labels': labels,
        'categories': categories,
        'sample_ids': sample_ids,
        'pair_ids': pair_ids,
    }


def plot_embeddings(
    data: Dict[str, Any],
    output_path: str,
    title: str = "Embedding Space",
    method: str = "umap",
    perplexity: int = 30,
    n_neighbors: int = 15,
):
    """Create 2D visualization of embeddings using specified method.
    
    Args:
        method: One of 'umap', 'tsne', or 'pca'
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib or sklearn not available")
        return
    
    embeddings = data['embeddings']
    categories = data['categories']
    
    if len(embeddings) < 5:
        logger.warning("Not enough embeddings for dimensionality reduction")
        return
    
    # Choose dimensionality reduction method
    if method == "umap":
        try:
            import umap
            logger.info(f"Computing UMAP with n_neighbors={n_neighbors}...")
            n_neighbors = min(n_neighbors, len(embeddings) - 1)
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, metric='cosine')
            embeddings_2d = reducer.fit_transform(embeddings)
            xlabel, ylabel = "UMAP 1", "UMAP 2"
        except ImportError:
            logger.warning("umap not available, falling back to t-SNE")
            method = "tsne"
    
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(perplexity, len(embeddings) - 1)
        logger.info(f"Computing t-SNE with perplexity={perplexity}...")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings)
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    
    if method == "pca":
        logger.info("Computing PCA...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    
    # Create color map
    unique_categories = list(set(categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for cat in unique_categories:
        mask = [c == cat for c in categories]
        points = embeddings_2d[mask]
        ax.scatter(
            points[:, 0], points[:, 1],
            c=[color_map[cat]],
            label=cat,
            alpha=0.7,
            s=50,
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({method.upper()})")
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved {method.upper()} plot to {output_path}")


def plot_similarity_heatmap(
    extractor: EmbeddingExtractor,
    sample: EmbeddingSample,
    output_path: str,
):
    """Create similarity heatmap for a single sample."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib or seaborn not available")
        return
    
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    # Collect all text variants
    texts = []
    text_labels = []
    
    # Add components
    for comp in sample.components[:3]:
        texts.append(comp)
        text_labels.append(f"C: {comp[:20]}...")
        
        negs = sample.component_negatives.get(comp, [])
        for neg in negs[:1]:
            neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
            if neg_text:
                texts.append(neg_text)
                text_labels.append(f"C-: {neg_text[:20]}...")
    
    # Add relations
    for rel in sample.relations[:2]:
        rel_text = f"{rel.get('subject', '')} {rel.get('relation_type', '')} {rel.get('object', '')}"
        if rel_text.strip():
            texts.append(rel_text)
            text_labels.append(f"R: {rel_text[:25]}...")
    
    # Add binding negatives
    for binding in sample.binding_negatives[:2]:
        bind_neg = binding.get('binding_neg_1', '')
        if bind_neg:
            texts.append(bind_neg)
            text_labels.append(f"B-: {bind_neg[:20]}...")
    
    if len(texts) < 2:
        return
    
    # Compute similarity matrix
    image_emb, text_embs, sim_matrix = extractor.compute_similarity_matrix(image, texts)
    
    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Image with thumbnail
    ax = axes[0]
    ax.imshow(image)
    ax.set_title(f"Sample: {sample.sample_id[:30]}")
    ax.axis('off')
    
    # Right: Similarity heatmap
    ax = axes[1]
    
    row_labels = ["Image"] + text_labels
    col_labels = text_labels
    
    sns.heatmap(
        sim_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Similarity Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity heatmap to {output_path}")


def plot_embedding_trajectories(
    extractor: EmbeddingExtractor,
    samples: List[EmbeddingSample],
    output_path: str,
    num_samples: int = 10,
):
    """Plot embedding trajectories: component -> relation -> full caption.
    
    Shows how embeddings evolve as we add more information.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib or sklearn not available")
        return
    
    all_embeddings = []
    all_labels = []
    all_stages = []  # 0=component, 1=relation, 2=full
    trajectory_ids = []
    
    traj_id = 0
    
    for sample in samples[:num_samples]:
        if not sample.relations:
            continue
        
        # Get first relation
        rel = sample.relations[0]
        subject = rel.get('subject', '')
        relation_type = rel.get('relation_type', '')
        obj = rel.get('object', '')
        
        if not subject or not relation_type or not obj:
            continue
        
        # Stage 0: Subject only
        subj_emb = extractor.encode_text(subject)
        all_embeddings.append(subj_emb)
        all_labels.append(subject)
        all_stages.append(0)
        trajectory_ids.append(traj_id)
        
        # Stage 1: Full relation
        rel_text = f"{subject} {relation_type} {obj}"
        rel_emb = extractor.encode_text(rel_text)
        all_embeddings.append(rel_emb)
        all_labels.append(rel_text)
        all_stages.append(1)
        trajectory_ids.append(traj_id)
        
        # Stage 2: Full caption (if different)
        if sample.original_caption and sample.original_caption != rel_text:
            full_emb = extractor.encode_text(sample.original_caption)
            all_embeddings.append(full_emb)
            all_labels.append(sample.original_caption[:50])
            all_stages.append(2)
            trajectory_ids.append(traj_id)
        
        traj_id += 1
        
        if traj_id >= num_samples:
            break
    
    if len(all_embeddings) < 5:
        logger.warning("Not enough trajectory data")
        return
    
    # Reduce to 2D with PCA
    embeddings = np.array(all_embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, traj_id))
    stage_markers = {0: 'o', 1: 's', 2: '^'}
    stage_labels = {0: 'Component', 1: 'Relation', 2: 'Full Caption'}
    
    # Plot points
    for stage in [0, 1, 2]:
        mask = [s == stage for s in all_stages]
        if any(mask):
            points = embeddings_2d[mask]
            point_colors = [colors[trajectory_ids[i]] for i, m in enumerate(mask) if m]
            ax.scatter(
                points[:, 0], points[:, 1],
                c=point_colors,
                marker=stage_markers[stage],
                s=100,
                label=stage_labels[stage],
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
            )
    
    # Draw trajectory lines
    for t_id in range(traj_id):
        t_mask = [tid == t_id for tid in trajectory_ids]
        t_points = embeddings_2d[t_mask]
        if len(t_points) >= 2:
            ax.plot(
                t_points[:, 0], t_points[:, 1],
                c=colors[t_id],
                alpha=0.5,
                linewidth=1,
                linestyle='--',
            )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Embedding Trajectories: Component → Relation → Full Caption")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved trajectory plot to {output_path}")


def plot_binding_pairs(
    data: Dict[str, Any],
    output_path: str,
):
    """Visualize binding pairs and their swaps in 2D."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib or sklearn not available")
        return
    
    embeddings = data['embeddings']
    categories = data['categories']
    pair_ids = data['pair_ids']
    labels = data['labels']
    
    if len(embeddings) < 4:
        logger.warning("Not enough binding pair data")
        return
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by category
    category_colors = {
        'original_1': 'blue',
        'original_2': 'green',
        'binding_neg_1': 'red',
        'binding_neg_2': 'orange',
    }
    
    # Plot points
    for cat, color in category_colors.items():
        mask = [c == cat for c in categories]
        if any(mask):
            points = embeddings_2d[mask]
            cat_labels = [labels[i] for i, m in enumerate(mask) if m]
            ax.scatter(
                points[:, 0], points[:, 1],
                c=color,
                label=cat.replace('_', ' ').title(),
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
            )
            
            # Add labels for first few
            for i, (x, y) in enumerate(points[:5]):
                ax.annotate(
                    cat_labels[i][:15],
                    (x, y),
                    fontsize=6,
                    alpha=0.7,
                )
    
    # Draw lines connecting binding pairs
    unique_pairs = list(set(pair_ids))
    for pid in unique_pairs[:20]:  # Limit to first 20 pairs
        pair_mask = [p == pid for p in pair_ids]
        pair_points = embeddings_2d[pair_mask]
        pair_cats = [categories[i] for i, m in enumerate(pair_mask) if m]
        
        # Connect original_1 to binding_neg_1 (they share attributes)
        # Connect original_2 to binding_neg_2
        if len(pair_points) == 4:
            # Original 1 -> Binding neg 1 (same attributes, different noun)
            ax.plot(
                [pair_points[0, 0], pair_points[2, 0]],
                [pair_points[0, 1], pair_points[2, 1]],
                'r--', alpha=0.3, linewidth=1,
            )
            # Original 2 -> Binding neg 2
            ax.plot(
                [pair_points[1, 0], pair_points[3, 0]],
                [pair_points[1, 1], pair_points[3, 1]],
                'orange', linestyle='--', alpha=0.3, linewidth=1,
            )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Binding Pairs: Attribute-Noun Swaps")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved binding pairs plot to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Embedding Visualization for Components and Relations")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "huggingface", "tripletclip", "external", "dac", "clove"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--force_openclip", action="store_true")
    parser.add_argument("--pretrained", type=str, default="openai")
    
    # Data arguments
    parser.add_argument("--json_folder", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="embedding_viz_results")
    parser.add_argument("--num_heatmaps", type=int, default=10)
    
    # Visualization arguments
    parser.add_argument("--method", type=str, default="umap",
                       choices=["umap", "tsne", "pca"],
                       help="Dimensionality reduction method for visualizations")
    parser.add_argument("--interactive", action="store_true",
                       help="Generate interactive HTML plots with plotly")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Initialize extractor
    extractor = EmbeddingExtractor(
        model_name=args.model_name,
        checkpoint_type=args.checkpoint_type,
        checkpoint_path=args.checkpoint_path,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
    )
    
    # 1. Component embeddings
    logger.info(f"\n=== Component Embeddings ({args.method.upper()}) ===")
    comp_data = create_component_embedding_data(extractor, samples)
    if len(comp_data['embeddings']) > 0:
        plot_embeddings(
            comp_data,
            str(output_dir / f"component_embeddings_{args.method}.png"),
            title="Component Embeddings: Positives vs Negatives",
            method=args.method,
        )
        
        # Save data
        np.save(output_dir / "component_embeddings.npy", comp_data['embeddings'])
        with open(output_dir / "component_metadata.json", 'w') as f:
            json.dump({
                'labels': comp_data['labels'],
                'categories': comp_data['categories'],
                'sample_ids': comp_data['sample_ids'],
            }, f, indent=2)
    
    # 2. Relation embeddings
    logger.info(f"\n=== Relation Embeddings ({args.method.upper()}) ===")
    rel_data = create_relation_embedding_data(extractor, samples)
    if len(rel_data['embeddings']) > 0:
        plot_embeddings(
            rel_data,
            str(output_dir / f"relation_embeddings_{args.method}.png"),
            title="Relation Embeddings: Original vs Swapped vs Partial",
            method=args.method,
        )
    
    # 3. Binding pairs visualization
    logger.info("\n=== Binding Pairs ===")
    binding_data = create_binding_embedding_data(extractor, samples)
    if len(binding_data['embeddings']) > 0:
        plot_binding_pairs(
            binding_data,
            str(output_dir / "binding_pairs.png"),
        )
    
    # 4. Embedding trajectories
    logger.info("\n=== Embedding Trajectories ===")
    plot_embedding_trajectories(
        extractor,
        samples,
        str(output_dir / "embedding_trajectories.png"),
        num_samples=15,
    )
    
    # 5. Individual similarity heatmaps
    logger.info("\n=== Similarity Heatmaps ===")
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(samples[:args.num_heatmaps]):
        plot_similarity_heatmap(
            extractor,
            sample,
            str(heatmap_dir / f"heatmap_{i:03d}_{sample.sample_id[:20]}.png"),
        )
    
    logger.info(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
