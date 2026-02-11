#!/usr/bin/env python3
"""
Embedding Visualization v2: Comparing CLIP vs Fine-tuned Models

Tells the story of how fine-tuning shifts embeddings:
1. Shows IMAGE embeddings alongside text embeddings
2. Visualizes similarity structure: image ↔ components ↔ negatives
3. Side-by-side model comparison (CLIP baseline vs fine-tuned)
4. Random sample inspection with actual images

Key Visualizations:
- Joint image-text embedding space (image point + text clusters around it)
- Similarity radar charts for individual samples
- Model comparison: CLIP vs Ours side-by-side
- Gallery view with sample images and their embedding neighborhoods

Usage:
    # Single model visualization
    python experiments/embedding_visualization_v2.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --output_dir embedding_viz_results \
        --num_samples 50

    # Model comparison (CLIP vs fine-tuned)
    python experiments/embedding_visualization_v2.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --compare_checkpoint /path/to/finetuned.pt \
        --compare_checkpoint_type external \
        --output_dir embedding_comparison_results
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Style settings for publication-quality plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for consistent visualization
COLORS = {
    'image': '#2E86AB',        # Blue for image embeddings
    'component_pos': '#28A745',  # Green for positive components
    'component_neg': '#DC3545',  # Red for negative components
    'relation_correct': '#6F42C1',  # Purple for correct relations
    'relation_swapped': '#FD7E14',  # Orange for swapped relations
    'binding_pos': '#17A2B8',   # Cyan for binding positives
    'binding_neg': '#E83E8C',   # Pink for binding negatives
    'caption': '#343A40',       # Dark gray for full captions
}

MARKERS = {
    'image': 's',           # Square for images
    'component_pos': 'o',   # Circle for components
    'component_neg': 'x',   # X for negatives
    'relation_correct': '^', # Triangle up
    'relation_swapped': 'v', # Triangle down
    'binding_pos': 'D',     # Diamond
    'binding_neg': 'd',     # Small diamond
}


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
# Model Wrapper
# =============================================================================

class EmbeddingModel:
    """Wrapper for CLIP-style models."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "openclip",
        force_openclip: bool = False,
        pretrained: str = "openai",
        name: str = "Model",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.name = name
        
        from utils.checkpoint_loader import load_checkpoint_model
        
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
        logger.info(f"Loaded {name}: {model_name}")
    
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
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts."""
        return np.array([self.encode_text(t) for t in texts])
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(emb1, emb2))


# =============================================================================
# Data Loading
# =============================================================================

def load_samples(
    json_folder: str,
    image_root: str,
    max_samples: int = 50,
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
# Embedding Collection
# =============================================================================

def collect_sample_embeddings(
    model: EmbeddingModel,
    sample: Sample,
    max_components: int = 3,
    max_negatives: int = 2,
) -> Dict[str, Any]:
    """Collect all embeddings for a sample."""
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return None
    
    result = {
        'sample_id': sample.sample_id,
        'image_path': sample.image_path,
        'image': image,
        'image_emb': model.encode_image(image),
        'texts': [],
        'text_embs': [],
        'categories': [],
        'is_positive': [],
    }
    
    # Components (positive)
    for comp in sample.components[:max_components]:
        result['texts'].append(comp)
        result['text_embs'].append(model.encode_text(comp))
        result['categories'].append('component_pos')
        result['is_positive'].append(True)
        
        # Component negatives
        negs = sample.component_negatives.get(comp, [])
        for neg in negs[:max_negatives]:
            neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
            if neg_text:
                result['texts'].append(neg_text)
                result['text_embs'].append(model.encode_text(neg_text))
                result['categories'].append('component_neg')
                result['is_positive'].append(False)
    
    # Relations
    for rel in sample.relations[:2]:
        subject = rel.get('subject', '')
        relation_type = rel.get('relation_type', '')
        obj = rel.get('object', '')
        
        if subject and relation_type and obj:
            # Correct relation
            rel_text = f"{subject} {relation_type} {obj}"
            result['texts'].append(rel_text)
            result['text_embs'].append(model.encode_text(rel_text))
            result['categories'].append('relation_correct')
            result['is_positive'].append(True)
            
            # Swapped relation
            swapped = f"{obj} {relation_type} {subject}"
            result['texts'].append(swapped)
            result['text_embs'].append(model.encode_text(swapped))
            result['categories'].append('relation_swapped')
            result['is_positive'].append(False)
    
    # Binding pairs
    for binding in sample.binding_negatives[:2]:
        comp1 = binding.get('component_1', '')
        bind_neg1 = binding.get('binding_neg_1', '')
        
        if comp1 and bind_neg1:
            result['texts'].append(comp1)
            result['text_embs'].append(model.encode_text(comp1))
            result['categories'].append('binding_pos')
            result['is_positive'].append(True)
            
            result['texts'].append(bind_neg1)
            result['text_embs'].append(model.encode_text(bind_neg1))
            result['categories'].append('binding_neg')
            result['is_positive'].append(False)
    
    result['text_embs'] = np.array(result['text_embs'])
    
    # Compute similarities
    result['similarities'] = np.array([
        model.similarity(result['image_emb'], emb) 
        for emb in result['text_embs']
    ])
    
    return result


def collect_all_embeddings(
    model: EmbeddingModel,
    samples: List[Sample],
) -> Dict[str, Any]:
    """Collect embeddings for all samples."""
    all_image_embs = []
    all_text_embs = []
    all_categories = []
    all_is_positive = []
    all_texts = []
    all_sample_ids = []
    all_images = []
    all_image_paths = []
    
    for sample in tqdm(samples, desc=f"Encoding with {model.name}"):
        data = collect_sample_embeddings(model, sample)
        if data is None:
            continue
        
        all_image_embs.append(data['image_emb'])
        all_images.append(data['image'])
        all_image_paths.append(data['image_path'])
        
        for i, emb in enumerate(data['text_embs']):
            all_text_embs.append(emb)
            all_categories.append(data['categories'][i])
            all_is_positive.append(data['is_positive'][i])
            all_texts.append(data['texts'][i])
            all_sample_ids.append(data['sample_id'])
    
    return {
        'image_embs': np.array(all_image_embs),
        'text_embs': np.array(all_text_embs),
        'categories': all_categories,
        'is_positive': all_is_positive,
        'texts': all_texts,
        'sample_ids': all_sample_ids,
        'images': all_images,
        'image_paths': all_image_paths,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: int = 30,
) -> np.ndarray:
    """Reduce embeddings to 2D."""
    if len(embeddings) < 5:
        return None
    
    if method == "umap":
        try:
            import umap
            n_neighbors = min(n_neighbors, len(embeddings) - 1)
            reducer = umap.UMAP(
                n_components=2, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist,
                random_state=42, 
                metric='cosine'
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("UMAP not available, falling back to t-SNE")
            method = "tsne"
    
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(perplexity, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        return reducer.fit_transform(embeddings)
    
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        return reducer.fit_transform(embeddings)
    
    return None


def plot_joint_embedding_space(
    data: Dict[str, Any],
    output_path: str,
    method: str = "umap",
    title: str = "Joint Image-Text Embedding Space",
    show_legend: bool = True,
):
    """
    Plot image and text embeddings together in 2D.
    Images are shown as squares, texts as circles/crosses.
    """
    # Combine all embeddings
    all_embs = np.vstack([data['image_embs'], data['text_embs']])
    
    # Reduce to 2D
    embs_2d = reduce_dimensions(all_embs, method=method)
    if embs_2d is None:
        return
    
    n_images = len(data['image_embs'])
    image_2d = embs_2d[:n_images]
    text_2d = embs_2d[n_images:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot image embeddings
    ax.scatter(
        image_2d[:, 0], image_2d[:, 1],
        c=COLORS['image'], marker=MARKERS['image'],
        s=120, alpha=0.8, edgecolors='white', linewidths=1,
        label='Image embeddings', zorder=10,
    )
    
    # Plot text embeddings by category
    categories = list(set(data['categories']))
    for cat in categories:
        mask = [c == cat for c in data['categories']]
        points = text_2d[mask]
        
        ax.scatter(
            points[:, 0], points[:, 1],
            c=COLORS.get(cat, 'gray'),
            marker=MARKERS.get(cat, 'o'),
            s=60 if 'neg' in cat or 'swapped' in cat else 80,
            alpha=0.6 if 'neg' in cat or 'swapped' in cat else 0.8,
            label=cat.replace('_', ' ').title(),
            zorder=5,
        )
    
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved joint embedding plot to {output_path}")


def plot_sample_neighborhood(
    model: EmbeddingModel,
    sample: Sample,
    output_path: str,
    method: str = "umap",
):
    """
    Plot a single sample's embedding neighborhood with the actual image.
    Shows the image, its embedding, and surrounding text embeddings.
    """
    data = collect_sample_embeddings(model, sample)
    if data is None:
        return
    
    # Create figure with image on left, embedding plot on right
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 3, width_ratios=[1, 2, 1], wspace=0.3)
    
    # Left: Image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(data['image'])
    ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Middle: Embedding space
    ax_emb = fig.add_subplot(gs[1])
    
    # Combine image + text embeddings
    all_embs = np.vstack([[data['image_emb']], data['text_embs']])
    
    # Reduce to 2D
    if len(all_embs) > 4:
        embs_2d = reduce_dimensions(all_embs, method=method)
        if embs_2d is None:
            plt.close()
            return
    else:
        # Use PCA for small samples
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(all_embs)
    
    image_2d = embs_2d[0]
    text_2d = embs_2d[1:]
    
    # Plot image embedding (center, with marker)
    ax_emb.scatter(
        [image_2d[0]], [image_2d[1]],
        c=COLORS['image'], marker='s', s=300,
        edgecolors='black', linewidths=2,
        label='Image', zorder=20,
    )
    
    # Add small image thumbnail at the point
    try:
        thumb = data['image'].copy()
        thumb.thumbnail((50, 50))
        imagebox = OffsetImage(thumb, zoom=0.8)
        ab = AnnotationBbox(imagebox, image_2d, frameon=True, 
                           bboxprops=dict(edgecolor='black', linewidth=1))
        ax_emb.add_artist(ab)
    except Exception:
        pass
    
    # Plot text embeddings with annotations
    for i, (cat, text, sim) in enumerate(zip(data['categories'], data['texts'], data['similarities'])):
        point = text_2d[i]
        
        ax_emb.scatter(
            [point[0]], [point[1]],
            c=COLORS.get(cat, 'gray'),
            marker=MARKERS.get(cat, 'o'),
            s=100, alpha=0.8,
            edgecolors='white', linewidths=0.5,
            zorder=10,
        )
        
        # Draw line from image to text (thicker for higher similarity)
        ax_emb.plot(
            [image_2d[0], point[0]], [image_2d[1], point[1]],
            color=COLORS.get(cat, 'gray'),
            alpha=0.3 + 0.5 * sim,
            linewidth=1 + 2 * sim,
            linestyle='-' if data['is_positive'][i] else '--',
            zorder=1,
        )
        
        # Add text label
        short_text = text[:25] + "..." if len(text) > 25 else text
        ax_emb.annotate(
            f"{short_text}\n(sim={sim:.2f})",
            point, fontsize=7,
            xytext=(5, 5), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )
    
    ax_emb.set_title(f"Embedding Neighborhood ({method.upper()})", fontsize=12, fontweight='bold')
    ax_emb.set_xlabel(f"{method.upper()} Dim 1")
    ax_emb.set_ylabel(f"{method.upper()} Dim 2")
    ax_emb.grid(True, alpha=0.3)
    
    # Right: Similarity bar chart
    ax_sim = fig.add_subplot(gs[2])
    
    # Sort by similarity
    sorted_idx = np.argsort(data['similarities'])[::-1]
    
    bar_colors = [COLORS.get(data['categories'][i], 'gray') for i in sorted_idx]
    bar_labels = [data['texts'][i][:20] + "..." if len(data['texts'][i]) > 20 else data['texts'][i] 
                  for i in sorted_idx]
    bar_values = [data['similarities'][i] for i in sorted_idx]
    
    y_pos = np.arange(len(bar_values))
    bars = ax_sim.barh(y_pos, bar_values, color=bar_colors, alpha=0.8)
    ax_sim.set_yticks(y_pos)
    ax_sim.set_yticklabels(bar_labels, fontsize=8)
    ax_sim.set_xlabel("Similarity to Image")
    ax_sim.set_title("Similarity Ranking", fontsize=12, fontweight='bold')
    ax_sim.set_xlim(0, 1)
    ax_sim.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, bar_values):
        ax_sim.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', va='center', fontsize=8)
    
    plt.suptitle(f"Sample: {sample.sample_id}", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved sample neighborhood plot to {output_path}")


def plot_model_comparison(
    model_baseline: EmbeddingModel,
    model_finetuned: EmbeddingModel,
    sample: Sample,
    output_path: str,
):
    """
    Side-by-side comparison of embedding neighborhoods for two models.
    Shows how fine-tuning shifts the relative positions of embeddings.
    """
    data_base = collect_sample_embeddings(model_baseline, sample)
    data_fine = collect_sample_embeddings(model_finetuned, sample)
    
    if data_base is None or data_fine is None:
        return
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, height_ratios=[1.5, 1], width_ratios=[1, 1, 0.8], wspace=0.25, hspace=0.3)
    
    # Top left: Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(data_base['image'])
    ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
    ax_img.axis('off')
    
    # Reduce both to same space for comparison
    all_embs_base = np.vstack([[data_base['image_emb']], data_base['text_embs']])
    all_embs_fine = np.vstack([[data_fine['image_emb']], data_fine['text_embs']])
    
    # Use PCA for consistent comparison
    from sklearn.decomposition import PCA
    
    def plot_embedding_subplot(ax, all_embs, data, title, model_name):
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(all_embs)
        image_2d = embs_2d[0]
        text_2d = embs_2d[1:]
        
        # Plot image
        ax.scatter([image_2d[0]], [image_2d[1]], c=COLORS['image'], marker='s', s=200,
                  edgecolors='black', linewidths=2, label='Image', zorder=20)
        
        # Plot texts with lines
        for i, (cat, text, sim) in enumerate(zip(data['categories'], data['texts'], data['similarities'])):
            point = text_2d[i]
            is_pos = data['is_positive'][i]
            
            ax.scatter([point[0]], [point[1]], c=COLORS.get(cat, 'gray'),
                      marker=MARKERS.get(cat, 'o'), s=80, alpha=0.8, zorder=10)
            
            ax.plot([image_2d[0], point[0]], [image_2d[1], point[1]],
                   color=COLORS.get(cat, 'gray'), alpha=0.2 + 0.4 * sim,
                   linewidth=1 + sim, linestyle='-' if is_pos else '--', zorder=1)
        
        ax.set_title(f"{model_name}\n{title}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    
    # Top middle: Baseline embeddings
    ax_base = fig.add_subplot(gs[0, 1])
    plot_embedding_subplot(ax_base, all_embs_base, data_base, "Embedding Space", model_baseline.name)
    
    # Top right: Fine-tuned embeddings  
    ax_fine = fig.add_subplot(gs[0, 2])
    plot_embedding_subplot(ax_fine, all_embs_fine, data_fine, "Embedding Space", model_finetuned.name)
    
    # Bottom: Similarity comparison bars
    ax_sim = fig.add_subplot(gs[1, :])
    
    n_texts = len(data_base['texts'])
    x = np.arange(n_texts)
    width = 0.35
    
    bars1 = ax_sim.bar(x - width/2, data_base['similarities'], width, 
                       label=model_baseline.name, color='#6C757D', alpha=0.7)
    bars2 = ax_sim.bar(x + width/2, data_fine['similarities'], width,
                       label=model_finetuned.name, color='#28A745', alpha=0.7)
    
    # Color bars based on positive/negative
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        if not data_base['is_positive'][i]:
            b1.set_facecolor('#DC3545')
            b1.set_alpha(0.5)
            b2.set_facecolor('#FF6B6B')
            b2.set_alpha(0.5)
    
    ax_sim.set_ylabel('Similarity to Image')
    ax_sim.set_title('Similarity Comparison: Baseline vs Fine-tuned', fontsize=12, fontweight='bold')
    ax_sim.set_xticks(x)
    
    # Truncate labels
    labels = [t[:15] + "..." if len(t) > 15 else t for t in data_base['texts']]
    ax_sim.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax_sim.legend()
    ax_sim.set_ylim(0, 1)
    ax_sim.grid(True, alpha=0.3, axis='y')
    
    # Add delta annotations
    for i, (s1, s2) in enumerate(zip(data_base['similarities'], data_fine['similarities'])):
        delta = s2 - s1
        color = 'green' if (delta > 0 and data_base['is_positive'][i]) or \
                          (delta < 0 and not data_base['is_positive'][i]) else 'red'
        ax_sim.annotate(f'{delta:+.2f}', (i, max(s1, s2) + 0.02), 
                       ha='center', fontsize=7, color=color)
    
    plt.suptitle(f"Model Comparison: {sample.sample_id}", fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved model comparison to {output_path}")


def plot_similarity_distribution(
    data: Dict[str, Any],
    model: EmbeddingModel,
    samples: List[Sample],
    output_path: str,
):
    """
    Plot distribution of similarities by category.
    Shows violin/box plots for positives vs negatives.
    """
    # Collect similarities by category
    cat_sims = {}
    
    for sample in tqdm(samples, desc="Computing similarities"):
        sample_data = collect_sample_embeddings(model, sample)
        if sample_data is None:
            continue
        
        for cat, sim, is_pos in zip(sample_data['categories'], 
                                     sample_data['similarities'],
                                     sample_data['is_positive']):
            key = cat
            if key not in cat_sims:
                cat_sims[key] = []
            cat_sims[key].append(sim)
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort categories: positives first, then negatives
    pos_cats = [c for c in cat_sims if 'pos' in c or 'correct' in c]
    neg_cats = [c for c in cat_sims if 'neg' in c or 'swapped' in c]
    sorted_cats = pos_cats + neg_cats
    
    positions = range(len(sorted_cats))
    data_to_plot = [cat_sims[c] for c in sorted_cats]
    colors = [COLORS.get(c, 'gray') for c in sorted_cats]
    
    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([c.replace('_', '\n') for c in sorted_cats], fontsize=9)
    ax.set_ylabel('Similarity to Image')
    ax.set_title(f'Similarity Distribution by Category ({model.name})', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    for i, data_cat in enumerate(data_to_plot):
        mean_val = np.mean(data_cat)
        ax.annotate(f'μ={mean_val:.2f}', (i, mean_val + 0.05), 
                   ha='center', fontsize=8, fontweight='bold')
    
    # Add dividing line between positives and negatives
    ax.axvline(x=len(pos_cats) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(pos_cats)/2 - 0.5, 0.98, 'POSITIVES', ha='center', fontsize=10, 
           color='green', fontweight='bold')
    ax.text(len(pos_cats) + len(neg_cats)/2 - 0.5, 0.98, 'NEGATIVES', ha='center', 
           fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved similarity distribution to {output_path}")


def plot_gallery_with_embeddings(
    model: EmbeddingModel,
    samples: List[Sample],
    output_path: str,
    n_samples: int = 9,
    method: str = "pca",
):
    """
    Gallery view: grid of images with their embedding neighborhoods.
    Shows random samples with image thumbnails and mini embedding plots.
    """
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, 4 * n_rows))
    
    selected = random.sample(samples, min(n_samples, len(samples)))
    
    for idx, sample in enumerate(selected):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        
        data = collect_sample_embeddings(model, sample)
        if data is None:
            continue
        
        # Image
        ax_img = axes[row, col] if n_rows > 1 else axes[col]
        ax_img.imshow(data['image'])
        ax_img.set_title(f"ID: {sample.sample_id[:15]}...", fontsize=9)
        ax_img.axis('off')
        
        # Mini embedding plot
        ax_emb = axes[row, col + 1] if n_rows > 1 else axes[col + 1]
        
        all_embs = np.vstack([[data['image_emb']], data['text_embs']])
        
        from sklearn.decomposition import PCA
        if len(all_embs) > 2:
            pca = PCA(n_components=2)
            embs_2d = pca.fit_transform(all_embs)
        else:
            continue
        
        image_2d = embs_2d[0]
        text_2d = embs_2d[1:]
        
        # Plot image point
        ax_emb.scatter([image_2d[0]], [image_2d[1]], c=COLORS['image'], marker='s', s=150,
                      edgecolors='black', linewidths=1.5, zorder=20)
        
        # Plot text points
        for i, cat in enumerate(data['categories']):
            ax_emb.scatter([text_2d[i, 0]], [text_2d[i, 1]], 
                          c=COLORS.get(cat, 'gray'),
                          marker=MARKERS.get(cat, 'o'),
                          s=40, alpha=0.7, zorder=10)
            
            # Draw connection line
            ax_emb.plot([image_2d[0], text_2d[i, 0]], [image_2d[1], text_2d[i, 1]],
                       color=COLORS.get(cat, 'gray'), alpha=0.3, linewidth=0.8)
        
        ax_emb.set_title(f"Embedding Space", fontsize=9)
        ax_emb.grid(True, alpha=0.3)
        ax_emb.set_xticks([])
        ax_emb.set_yticks([])
    
    # Hide unused axes
    for idx in range(len(selected), n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        if n_rows > 1:
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
        else:
            axes[col].axis('off')
            axes[col + 1].axis('off')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['image'], label='Image'),
        mpatches.Patch(facecolor=COLORS['component_pos'], label='Component (+)'),
        mpatches.Patch(facecolor=COLORS['component_neg'], label='Component (-)'),
        mpatches.Patch(facecolor=COLORS['relation_correct'], label='Relation (+)'),
        mpatches.Patch(facecolor=COLORS['relation_swapped'], label='Relation (swapped)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
    
    plt.suptitle(f"Sample Gallery with Embedding Neighborhoods ({model.name})", 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved gallery to {output_path}")


def plot_accuracy_by_margin(
    model: EmbeddingModel,
    samples: List[Sample],
    output_path: str,
):
    """
    Plot accuracy vs similarity margin.
    Shows how often positive > negative and by how much.
    """
    margins = {
        'component': [],
        'relation': [],
        'binding': [],
    }
    
    for sample in tqdm(samples, desc="Computing margins"):
        data = collect_sample_embeddings(model, sample)
        if data is None:
            continue
        
        # Group by type
        comp_pos = [(s, t) for s, t, c, p in zip(data['similarities'], data['texts'], 
                                                  data['categories'], data['is_positive']) 
                   if c == 'component_pos']
        comp_neg = [(s, t) for s, t, c, p in zip(data['similarities'], data['texts'],
                                                  data['categories'], data['is_positive'])
                   if c == 'component_neg']
        
        for pos_sim, _ in comp_pos:
            for neg_sim, _ in comp_neg:
                margins['component'].append(pos_sim - neg_sim)
        
        rel_correct = [(s, t) for s, t, c in zip(data['similarities'], data['texts'], data['categories'])
                      if c == 'relation_correct']
        rel_swapped = [(s, t) for s, t, c in zip(data['similarities'], data['texts'], data['categories'])
                      if c == 'relation_swapped']
        
        for (pos_sim, _), (neg_sim, _) in zip(rel_correct, rel_swapped):
            margins['relation'].append(pos_sim - neg_sim)
        
        bind_pos = [(s, t) for s, t, c in zip(data['similarities'], data['texts'], data['categories'])
                   if c == 'binding_pos']
        bind_neg = [(s, t) for s, t, c in zip(data['similarities'], data['texts'], data['categories'])
                   if c == 'binding_neg']
        
        for (pos_sim, _), (neg_sim, _) in zip(bind_pos, bind_neg):
            margins['binding'].append(pos_sim - neg_sim)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#28A745', '#6F42C1', '#17A2B8']
    labels = ['Components', 'Relations', 'Bindings']
    
    for i, (key, vals) in enumerate(margins.items()):
        if vals:
            vals = np.array(vals)
            acc = (vals > 0).mean()
            mean_margin = vals.mean()
            
            ax.hist(vals, bins=30, alpha=0.5, color=colors[i], label=f'{labels[i]} (acc={acc:.1%}, μ={mean_margin:.3f})')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    ax.set_xlabel('Similarity Margin (positive - negative)')
    ax.set_ylabel('Count')
    ax.set_title(f'Similarity Margin Distribution ({model.name})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved margin plot to {output_path}")


def compute_and_save_stats(
    model: EmbeddingModel,
    samples: List[Sample],
    output_path: str,
):
    """Compute and save summary statistics."""
    stats = {
        'model_name': model.name,
        'n_samples': len(samples),
        'component_accuracy': 0,
        'relation_accuracy': 0,
        'binding_accuracy': 0,
        'similarities': {},
    }
    
    comp_correct = 0
    comp_total = 0
    rel_correct = 0
    rel_total = 0
    bind_correct = 0
    bind_total = 0
    
    cat_sims = {}
    
    for sample in tqdm(samples, desc="Computing statistics"):
        data = collect_sample_embeddings(model, sample)
        if data is None:
            continue
        
        for cat, sim in zip(data['categories'], data['similarities']):
            if cat not in cat_sims:
                cat_sims[cat] = []
            cat_sims[cat].append(sim)
        
        # Component accuracy
        comp_pos = [s for s, c in zip(data['similarities'], data['categories']) if c == 'component_pos']
        comp_neg = [s for s, c in zip(data['similarities'], data['categories']) if c == 'component_neg']
        
        for pos_sim in comp_pos:
            for neg_sim in comp_neg:
                comp_total += 1
                if pos_sim > neg_sim:
                    comp_correct += 1
        
        # Relation accuracy
        rel_sims = {c: s for s, c in zip(data['similarities'], data['categories']) 
                   if c in ['relation_correct', 'relation_swapped']}
        if 'relation_correct' in rel_sims and 'relation_swapped' in rel_sims:
            rel_total += 1
            if rel_sims['relation_correct'] > rel_sims['relation_swapped']:
                rel_correct += 1
        
        # Binding accuracy
        bind_sims = [(s, c) for s, c in zip(data['similarities'], data['categories'])
                    if c in ['binding_pos', 'binding_neg']]
        bind_pos = [s for s, c in bind_sims if c == 'binding_pos']
        bind_neg = [s for s, c in bind_sims if c == 'binding_neg']
        for pos_sim, neg_sim in zip(bind_pos, bind_neg):
            bind_total += 1
            if pos_sim > neg_sim:
                bind_correct += 1
    
    stats['component_accuracy'] = comp_correct / max(1, comp_total)
    stats['relation_accuracy'] = rel_correct / max(1, rel_total)
    stats['binding_accuracy'] = bind_correct / max(1, bind_total)
    
    stats['similarities'] = {
        cat: {'mean': float(np.mean(sims)), 'std': float(np.std(sims)), 'n': len(sims)}
        for cat, sims in cat_sims.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"EMBEDDING STATISTICS: {model.name}")
    print("=" * 60)
    print(f"Samples analyzed: {len(samples)}")
    print(f"\nAccuracy (positive > negative):")
    print(f"  Components: {stats['component_accuracy']:.1%}")
    print(f"  Relations:  {stats['relation_accuracy']:.1%}")
    print(f"  Bindings:   {stats['binding_accuracy']:.1%}")
    print(f"\nMean Similarities:")
    for cat, info in stats['similarities'].items():
        print(f"  {cat}: {info['mean']:.3f} ± {info['std']:.3f}")
    print("=" * 60)
    
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Embedding Visualization v2")
    
    # Model arguments (baseline)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "huggingface", "tripletclip", "external", "dac", "clove"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--force_openclip", action="store_true")
    parser.add_argument("--pretrained", type=str, default="openai")
    
    # Comparison model arguments (for side-by-side)
    parser.add_argument("--compare_checkpoint", type=str, default=None,
                       help="Path to fine-tuned checkpoint for comparison")
    parser.add_argument("--compare_checkpoint_type", type=str, default="external")
    parser.add_argument("--compare_name", type=str, default="Fine-tuned",
                       help="Name for comparison model")
    
    # Data arguments
    parser.add_argument("--json_folder", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="embedding_viz_v2")
    parser.add_argument("--method", type=str, default="umap",
                       choices=["umap", "tsne", "pca"])
    parser.add_argument("--num_gallery", type=int, default=9)
    parser.add_argument("--num_neighborhood", type=int, default=5)
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neighborhoods").mkdir(exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    
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
    
    # Load baseline model
    model_baseline = EmbeddingModel(
        model_name=args.model_name,
        checkpoint_type=args.checkpoint_type,
        checkpoint_path=args.checkpoint_path,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
        name="CLIP (Baseline)",
    )
    
    # Collect embeddings
    logger.info("\n=== Collecting Baseline Embeddings ===")
    data_baseline = collect_all_embeddings(model_baseline, samples)
    
    # 1. Joint embedding space
    logger.info("\n=== Joint Embedding Space ===")
    plot_joint_embedding_space(
        data_baseline,
        str(output_dir / f"joint_embedding_space_{args.method}.png"),
        method=args.method,
        title=f"Joint Image-Text Embedding Space ({model_baseline.name})",
    )
    
    # 2. Similarity distribution
    logger.info("\n=== Similarity Distribution ===")
    plot_similarity_distribution(
        data_baseline,
        model_baseline,
        samples,
        str(output_dir / "similarity_distribution.png"),
    )
    
    # 3. Gallery view
    logger.info("\n=== Gallery View ===")
    plot_gallery_with_embeddings(
        model_baseline,
        samples,
        str(output_dir / "gallery.png"),
        n_samples=args.num_gallery,
    )
    
    # 4. Individual sample neighborhoods
    logger.info("\n=== Sample Neighborhoods ===")
    for i, sample in enumerate(samples[:args.num_neighborhood]):
        plot_sample_neighborhood(
            model_baseline,
            sample,
            str(output_dir / "neighborhoods" / f"sample_{i:02d}_{sample.sample_id[:20]}.png"),
            method=args.method,
        )
    
    # 5. Accuracy by margin
    logger.info("\n=== Accuracy by Margin ===")
    plot_accuracy_by_margin(
        model_baseline,
        samples,
        str(output_dir / "accuracy_margin.png"),
    )
    
    # 6. Statistics
    stats_baseline = compute_and_save_stats(
        model_baseline,
        samples,
        str(output_dir / "stats_baseline.json"),
    )
    
    # If comparison model provided
    if args.compare_checkpoint:
        logger.info("\n=== Loading Comparison Model ===")
        model_finetuned = EmbeddingModel(
            model_name=args.model_name,
            checkpoint_type=args.compare_checkpoint_type,
            checkpoint_path=args.compare_checkpoint,
            force_openclip=args.force_openclip,
            name=args.compare_name,
        )
        
        # Collect fine-tuned embeddings
        data_finetuned = collect_all_embeddings(model_finetuned, samples)
        
        # Joint space for fine-tuned
        plot_joint_embedding_space(
            data_finetuned,
            str(output_dir / f"joint_embedding_space_{args.method}_finetuned.png"),
            method=args.method,
            title=f"Joint Image-Text Embedding Space ({model_finetuned.name})",
        )
        
        # Similarity distribution for fine-tuned
        plot_similarity_distribution(
            data_finetuned,
            model_finetuned,
            samples,
            str(output_dir / "similarity_distribution_finetuned.png"),
        )
        
        # Model comparison plots
        logger.info("\n=== Model Comparisons ===")
        for i, sample in enumerate(samples[:args.num_neighborhood]):
            plot_model_comparison(
                model_baseline,
                model_finetuned,
                sample,
                str(output_dir / "comparisons" / f"comparison_{i:02d}_{sample.sample_id[:20]}.png"),
            )
        
        # Fine-tuned statistics
        stats_finetuned = compute_and_save_stats(
            model_finetuned,
            samples,
            str(output_dir / "stats_finetuned.json"),
        )
        
        # Print comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<25} {'Baseline':>15} {'Fine-tuned':>15} {'Delta':>10}")
        print("-" * 60)
        print(f"{'Component Accuracy':<25} {stats_baseline['component_accuracy']:>15.1%} "
              f"{stats_finetuned['component_accuracy']:>15.1%} "
              f"{stats_finetuned['component_accuracy'] - stats_baseline['component_accuracy']:>+10.1%}")
        print(f"{'Relation Accuracy':<25} {stats_baseline['relation_accuracy']:>15.1%} "
              f"{stats_finetuned['relation_accuracy']:>15.1%} "
              f"{stats_finetuned['relation_accuracy'] - stats_baseline['relation_accuracy']:>+10.1%}")
        print(f"{'Binding Accuracy':<25} {stats_baseline['binding_accuracy']:>15.1%} "
              f"{stats_finetuned['binding_accuracy']:>15.1%} "
              f"{stats_finetuned['binding_accuracy'] - stats_baseline['binding_accuracy']:>+10.1%}")
        print("=" * 60)
    
    logger.info(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
