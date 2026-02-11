#!/usr/bin/env python3
"""
Attention Visualization for CLIP Models

Visualizes attention patterns for:
1. Image attention to text tokens (which image regions attend to which words)
2. Text attention patterns for components, relations, half-truths
3. Cross-attention heatmaps overlaid on images
4. Attention difference maps (correct vs incorrect captions)

Uses gradient-weighted attention for better interpretability.

Usage:
    python experiments/attention_visualization.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --output_dir attention_viz_results \
        --num_samples 20
        
    # With fine-tuned model
    python experiments/attention_visualization.py \
        --checkpoint_path /path/to/checkpoint.pt \
        --checkpoint_type external \
        --output_dir attention_viz_finetuned
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
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
# Attention Extractor with Hooks
# =============================================================================

class AttentionExtractor:
    """Extracts attention maps from CLIP vision transformer."""
    
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
        
        # Import here to avoid circular imports
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
        self.model_name = model_name
        
        # Storage for attention maps
        self._attention_maps = []
        self._hooks = []
        
        # Detect model type and register hooks
        self._setup_attention_hooks()
        
        logger.info("Model loaded with attention hooks")
    
    def _setup_attention_hooks(self):
        """Register forward hooks to capture attention maps."""
        # Try to find the visual encoder
        visual = None
        
        if hasattr(self.model, 'visual'):
            visual = self.model.visual
        elif hasattr(self.model, 'vision_model'):
            visual = self.model.vision_model
        elif hasattr(self.model, 'image_encoder'):
            visual = self.model.image_encoder
        
        if visual is None:
            logger.warning("Could not find visual encoder, attention extraction may not work")
            return
        
        # Find transformer blocks
        transformer = None
        
        if hasattr(visual, 'transformer'):
            transformer = visual.transformer
        elif hasattr(visual, 'encoder'):
            transformer = visual.encoder
        elif hasattr(visual, 'blocks'):
            transformer = visual
        
        if transformer is None:
            logger.warning("Could not find transformer blocks")
            return
        
        # Find attention layers
        layers = None
        
        if hasattr(transformer, 'resblocks'):
            layers = transformer.resblocks
        elif hasattr(transformer, 'layers'):
            layers = transformer.layers
        elif hasattr(transformer, 'blocks'):
            layers = transformer.blocks
        
        if layers is None:
            logger.warning("Could not find attention layers")
            return
        
        # Register hooks on attention modules
        for i, layer in enumerate(layers):
            # Try different attention module names
            attn = None
            if hasattr(layer, 'attn'):
                attn = layer.attn
            elif hasattr(layer, 'self_attn'):
                attn = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn = layer.attention
            
            if attn is not None:
                hook = attn.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)
        
        logger.info(f"Registered {len(self._hooks)} attention hooks")
    
    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights."""
        # Try to extract attention weights from different output formats
        attn_weights = None
        
        if isinstance(output, tuple):
            # Some models return (output, attention_weights)
            for o in output:
                if isinstance(o, torch.Tensor) and len(o.shape) >= 3:
                    # Check if it looks like attention weights (square-ish)
                    if o.shape[-1] == o.shape[-2] or (len(o.shape) == 4 and o.shape[-1] == o.shape[-2]):
                        attn_weights = o
                        break
        
        if attn_weights is not None:
            self._attention_maps.append(attn_weights.detach().cpu())
    
    def clear_attention_maps(self):
        """Clear stored attention maps."""
        self._attention_maps = []
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """Get stored attention maps."""
        return self._attention_maps
    
    @torch.no_grad()
    def encode_image_with_attention(
        self,
        image: Image.Image,
    ) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """Encode image and return embedding + attention maps."""
        self.clear_attention_maps()
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        features = self.model.encode_image(image_input)
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy()[0], self.get_attention_maps()
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text."""
        try:
            text_input = self.tokenize([text]).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_input = self.tokenize([text], truncate=True).to(self.device)
        
        features = self.model.encode_text(text_input)
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()[0]
    
    def compute_similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """Compute similarity between embeddings."""
        return float(np.dot(image_emb, text_emb))
    
    def compute_grad_cam(
        self,
        image: Image.Image,
        text: str,
    ) -> Optional[np.ndarray]:
        """Compute GradCAM-style attention map for image-text pair.
        
        Returns:
            Attention heatmap of shape (H, W) or None if failed
        """
        self.model.zero_grad()
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_input.requires_grad = True
        
        try:
            text_input = self.tokenize([text]).to(self.device)
        except:
            text_input = self.tokenize([text], truncate=True).to(self.device)
        
        # Get features
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        similarity = (image_features * text_features).sum()
        
        # Backward pass
        similarity.backward()
        
        # Get gradients
        if image_input.grad is not None:
            gradients = image_input.grad.detach().cpu().numpy()[0]  # (C, H, W)
            
            # Compute gradient magnitude
            grad_magnitude = np.abs(gradients).mean(axis=0)  # (H, W)
            
            # Normalize to [0, 1]
            grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
            
            return grad_magnitude
        
        return None
    
    def aggregate_attention_to_spatial(
        self,
        attention_maps: List[torch.Tensor],
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 32,
    ) -> Optional[np.ndarray]:
        """Aggregate attention maps to spatial heatmap.
        
        Takes the last layer's attention from CLS token to patch tokens
        and reshapes to spatial grid.
        """
        if not attention_maps:
            return None
        
        # Get last layer attention
        last_attn = attention_maps[-1]  # Usually (1, num_heads, seq_len, seq_len)
        
        if len(last_attn.shape) == 4:
            # Average over heads
            attn = last_attn.mean(dim=1)[0]  # (seq_len, seq_len)
        elif len(last_attn.shape) == 3:
            attn = last_attn[0]  # (seq_len, seq_len)
        else:
            return None
        
        # Get attention from CLS token (first token) to patch tokens
        # Skip the CLS token itself
        num_patches = attn.shape[0] - 1
        
        if num_patches <= 0:
            return None
        
        cls_attn = attn[0, 1:].numpy()  # Attention from CLS to patches
        
        # Compute grid size
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            # Not a square grid, skip
            return None
        
        # Reshape to grid
        spatial_attn = cls_attn.reshape(grid_size, grid_size)
        
        # Resize to image size
        from PIL import Image as PILImage
        spatial_attn_img = PILImage.fromarray((spatial_attn * 255).astype(np.uint8))
        spatial_attn_img = spatial_attn_img.resize(image_size, PILImage.BILINEAR)
        
        return np.array(spatial_attn_img) / 255.0


# =============================================================================
# Sample Loader (same as embedding_visualization.py)
# =============================================================================

@dataclass
class AttentionSample:
    """Sample for attention visualization."""
    image_path: str
    sample_id: str
    original_caption: str
    components: List[str]
    component_negatives: Dict[str, List[Dict[str, str]]]
    relations: List[Dict[str, str]]
    binding_negatives: List[Dict[str, str]]


def load_samples(
    json_folder: str,
    image_root: str,
    max_samples: int = 20,
    seed: int = 42,
) -> List[AttentionSample]:
    """Load samples for attention visualization."""
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
        
        samples.append(AttentionSample(
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
# Visualization Functions
# =============================================================================

def plot_attention_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    title: str,
    output_path: str,
    alpha: float = 0.5,
):
    """Overlay attention heatmap on image."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Attention Heatmap")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size))
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=alpha)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_comparison(
    image: Image.Image,
    heatmaps: List[Tuple[str, np.ndarray, float]],  # [(label, heatmap, similarity), ...]
    output_path: str,
    suptitle: str = "Attention Comparison",
):
    """Compare attention heatmaps for different captions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return
    
    n_heatmaps = len(heatmaps)
    if n_heatmaps == 0:
        return
    
    n_cols = min(4, n_heatmaps + 1)  # +1 for original image
    n_rows = (n_heatmaps + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmaps
    for i, (label, heatmap, sim) in enumerate(heatmaps):
        ax = axes[i + 1]
        ax.imshow(image)
        
        if heatmap is not None:
            heatmap_resized = np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size)
            )
            ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
        
        ax.set_title(f"{label[:30]}...\nsim={sim:.3f}", fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(len(heatmaps) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_difference(
    image: Image.Image,
    heatmap_correct: np.ndarray,
    heatmap_incorrect: np.ndarray,
    label_correct: str,
    label_incorrect: str,
    sim_correct: float,
    sim_incorrect: float,
    output_path: str,
):
    """Show attention difference between correct and incorrect caption."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Correct caption attention
    axes[1].imshow(image)
    if heatmap_correct is not None:
        hm = np.array(Image.fromarray((heatmap_correct * 255).astype(np.uint8)).resize(image.size))
        axes[1].imshow(hm, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Correct: {label_correct[:25]}...\nsim={sim_correct:.3f}", fontsize=9)
    axes[1].axis('off')
    
    # Incorrect caption attention
    axes[2].imshow(image)
    if heatmap_incorrect is not None:
        hm = np.array(Image.fromarray((heatmap_incorrect * 255).astype(np.uint8)).resize(image.size))
        axes[2].imshow(hm, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Incorrect: {label_incorrect[:25]}...\nsim={sim_incorrect:.3f}", fontsize=9)
    axes[2].axis('off')
    
    # Difference map
    if heatmap_correct is not None and heatmap_incorrect is not None:
        diff = heatmap_correct - heatmap_incorrect
        # Normalize to [-1, 1] -> [0, 1] for visualization
        diff_norm = (diff + 1) / 2
        
        axes[3].imshow(image)
        hm = np.array(Image.fromarray((diff_norm * 255).astype(np.uint8)).resize(image.size))
        axes[3].imshow(hm, cmap='RdBu_r', alpha=0.6)
        axes[3].set_title(f"Difference (Correct - Incorrect)\nRed=Correct focus, Blue=Incorrect focus", fontsize=9)
    else:
        axes[3].set_title("Difference N/A")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_component_attention(
    extractor: AttentionExtractor,
    sample: AttentionSample,
    output_dir: Path,
):
    """Visualize attention for components and their negatives."""
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    # Get image embedding
    image_emb, attention_maps = extractor.encode_image_with_attention(image)
    
    heatmaps = []
    
    for comp in sample.components[:3]:
        # Positive component
        comp_emb = extractor.encode_text(comp)
        sim = extractor.compute_similarity(image_emb, comp_emb)
        
        # Get GradCAM
        heatmap = extractor.compute_grad_cam(image, comp)
        if heatmap is None:
            # Fallback to aggregated attention
            heatmap = extractor.aggregate_attention_to_spatial(attention_maps, image.size)
        
        heatmaps.append((f"✓ {comp}", heatmap, sim))
        
        # Negative components
        negs = sample.component_negatives.get(comp, [])
        for neg in negs[:2]:
            neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
            change_type = neg.get('change_type', '') if isinstance(neg, dict) else ''
            
            if not neg_text:
                continue
            
            neg_emb = extractor.encode_text(neg_text)
            neg_sim = extractor.compute_similarity(image_emb, neg_emb)
            
            neg_heatmap = extractor.compute_grad_cam(image, neg_text)
            if neg_heatmap is None:
                neg_heatmap = extractor.aggregate_attention_to_spatial(attention_maps, image.size)
            
            heatmaps.append((f"✗ {neg_text} ({change_type})", neg_heatmap, neg_sim))
    
    if heatmaps:
        plot_attention_comparison(
            image,
            heatmaps,
            str(output_dir / f"component_attention_{sample.sample_id[:20]}.png"),
            suptitle=f"Component Attention: {sample.sample_id}",
        )


def visualize_relation_attention(
    extractor: AttentionExtractor,
    sample: AttentionSample,
    output_dir: Path,
):
    """Visualize attention for relations."""
    if not sample.relations:
        return
    
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    image_emb, attention_maps = extractor.encode_image_with_attention(image)
    
    heatmaps = []
    
    for rel in sample.relations[:2]:
        subject = rel.get('subject', '')
        relation_type = rel.get('relation_type', '')
        obj = rel.get('object', '')
        
        if not subject or not relation_type or not obj:
            continue
        
        # Full relation
        rel_text = f"{subject} {relation_type} {obj}"
        rel_emb = extractor.encode_text(rel_text)
        rel_sim = extractor.compute_similarity(image_emb, rel_emb)
        rel_heatmap = extractor.compute_grad_cam(image, rel_text)
        heatmaps.append((f"✓ {rel_text}", rel_heatmap, rel_sim))
        
        # Subject only (partial)
        subj_emb = extractor.encode_text(subject)
        subj_sim = extractor.compute_similarity(image_emb, subj_emb)
        subj_heatmap = extractor.compute_grad_cam(image, subject)
        heatmaps.append((f"Partial: {subject}", subj_heatmap, subj_sim))
        
        # Swapped relation
        swapped = f"{obj} {relation_type} {subject}"
        swapped_emb = extractor.encode_text(swapped)
        swapped_sim = extractor.compute_similarity(image_emb, swapped_emb)
        swapped_heatmap = extractor.compute_grad_cam(image, swapped)
        heatmaps.append((f"✗ Swap: {swapped}", swapped_heatmap, swapped_sim))
        
        # Relation negatives from the relation itself
        rel_negs = rel.get('negatives', [])
        for neg in rel_negs[:1]:
            neg_rel = neg.get('relation_type', '')
            change_type = neg.get('change_type', '')
            if neg_rel:
                neg_text = f"{subject} {neg_rel} {obj}"
                neg_emb = extractor.encode_text(neg_text)
                neg_sim = extractor.compute_similarity(image_emb, neg_emb)
                neg_heatmap = extractor.compute_grad_cam(image, neg_text)
                heatmaps.append((f"✗ {change_type}: {neg_text}", neg_heatmap, neg_sim))
    
    if heatmaps:
        plot_attention_comparison(
            image,
            heatmaps,
            str(output_dir / f"relation_attention_{sample.sample_id[:20]}.png"),
            suptitle=f"Relation Attention: {sample.sample_id}",
        )


def visualize_half_truth_attention(
    extractor: AttentionExtractor,
    sample: AttentionSample,
    output_dir: Path,
):
    """Visualize attention for half-truth scenarios."""
    if len(sample.components) < 2:
        return
    
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    image_emb, _ = extractor.encode_image_with_attention(image)
    
    # Build half-truth captions
    comp_a = sample.components[0]
    comp_b = sample.components[1]
    
    # Get negative for comp_b
    negs_b = sample.component_negatives.get(comp_b, [])
    if not negs_b:
        return
    
    neg_b = negs_b[0].get('negative', '') if isinstance(negs_b[0], dict) else negs_b[0]
    if not neg_b:
        return
    
    # Captions
    short_correct = comp_a
    long_correct = f"{comp_a} and {comp_b}"
    long_incorrect = f"{comp_a} and {neg_b}"
    
    # Get embeddings and heatmaps
    short_emb = extractor.encode_text(short_correct)
    short_sim = extractor.compute_similarity(image_emb, short_emb)
    short_heatmap = extractor.compute_grad_cam(image, short_correct)
    
    long_correct_emb = extractor.encode_text(long_correct)
    long_correct_sim = extractor.compute_similarity(image_emb, long_correct_emb)
    long_correct_heatmap = extractor.compute_grad_cam(image, long_correct)
    
    long_incorrect_emb = extractor.encode_text(long_incorrect)
    long_incorrect_sim = extractor.compute_similarity(image_emb, long_incorrect_emb)
    long_incorrect_heatmap = extractor.compute_grad_cam(image, long_incorrect)
    
    # Plot comparison
    heatmaps = [
        (f"Short (correct): {short_correct}", short_heatmap, short_sim),
        (f"Long (correct): {long_correct}", long_correct_heatmap, long_correct_sim),
        (f"Long (incorrect): {long_incorrect}", long_incorrect_heatmap, long_incorrect_sim),
    ]
    
    plot_attention_comparison(
        image,
        heatmaps,
        str(output_dir / f"half_truth_attention_{sample.sample_id[:20]}.png"),
        suptitle=f"Half-Truth Attention: {sample.sample_id}",
    )
    
    # Also plot difference
    if long_correct_heatmap is not None and long_incorrect_heatmap is not None:
        plot_attention_difference(
            image,
            long_correct_heatmap,
            long_incorrect_heatmap,
            long_correct,
            long_incorrect,
            long_correct_sim,
            long_incorrect_sim,
            str(output_dir / f"half_truth_diff_{sample.sample_id[:20]}.png"),
        )


def visualize_binding_attention(
    extractor: AttentionExtractor,
    sample: AttentionSample,
    output_dir: Path,
):
    """Visualize attention for binding pairs (attribute-noun swaps)."""
    if not sample.binding_negatives:
        return
    
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    image_emb, _ = extractor.encode_image_with_attention(image)
    
    heatmaps = []
    
    for binding in sample.binding_negatives[:2]:
        comp1 = binding.get('component_1', '')
        comp2 = binding.get('component_2', '')
        bind_neg1 = binding.get('binding_neg_1', '')
        bind_neg2 = binding.get('binding_neg_2', '')
        
        if not comp1 or not bind_neg1:
            continue
        
        # Original component 1
        emb1 = extractor.encode_text(comp1)
        sim1 = extractor.compute_similarity(image_emb, emb1)
        heatmap1 = extractor.compute_grad_cam(image, comp1)
        heatmaps.append((f"✓ {comp1}", heatmap1, sim1))
        
        # Binding negative 1 (same attributes, wrong noun)
        neg_emb1 = extractor.encode_text(bind_neg1)
        neg_sim1 = extractor.compute_similarity(image_emb, neg_emb1)
        neg_heatmap1 = extractor.compute_grad_cam(image, bind_neg1)
        heatmaps.append((f"✗ Binding: {bind_neg1}", neg_heatmap1, neg_sim1))
        
        if comp2 and bind_neg2:
            # Original component 2
            emb2 = extractor.encode_text(comp2)
            sim2 = extractor.compute_similarity(image_emb, emb2)
            heatmap2 = extractor.compute_grad_cam(image, comp2)
            heatmaps.append((f"✓ {comp2}", heatmap2, sim2))
            
            # Binding negative 2
            neg_emb2 = extractor.encode_text(bind_neg2)
            neg_sim2 = extractor.compute_similarity(image_emb, neg_emb2)
            neg_heatmap2 = extractor.compute_grad_cam(image, bind_neg2)
            heatmaps.append((f"✗ Binding: {bind_neg2}", neg_heatmap2, neg_sim2))
    
    if heatmaps:
        plot_attention_comparison(
            image,
            heatmaps,
            str(output_dir / f"binding_attention_{sample.sample_id[:20]}.png"),
            suptitle=f"Binding Attention (Attribute-Noun Swaps): {sample.sample_id}",
        )


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_attention_statistics(
    extractor: AttentionExtractor,
    samples: List[AttentionSample],
    output_path: str,
):
    """Compute summary statistics about attention patterns."""
    stats = {
        'component': {'correct_higher': 0, 'incorrect_higher': 0, 'total': 0},
        'relation': {'correct_higher': 0, 'incorrect_higher': 0, 'total': 0},
        'binding': {'correct_higher': 0, 'incorrect_higher': 0, 'total': 0},
        'half_truth': {'short_correct_wins': 0, 'long_incorrect_wins': 0, 'total': 0},
    }
    
    similarities = {
        'component_positive': [],
        'component_negative': [],
        'relation_correct': [],
        'relation_swapped': [],
        'binding_correct': [],
        'binding_swapped': [],
    }
    
    for sample in tqdm(samples, desc="Computing attention statistics"):
        try:
            image = Image.open(sample.image_path).convert('RGB')
            image_emb, _ = extractor.encode_image_with_attention(image)
        except:
            continue
        
        # Component statistics
        for comp in sample.components[:3]:
            comp_emb = extractor.encode_text(comp)
            comp_sim = extractor.compute_similarity(image_emb, comp_emb)
            similarities['component_positive'].append(comp_sim)
            
            negs = sample.component_negatives.get(comp, [])
            for neg in negs[:2]:
                neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
                if neg_text:
                    neg_emb = extractor.encode_text(neg_text)
                    neg_sim = extractor.compute_similarity(image_emb, neg_emb)
                    similarities['component_negative'].append(neg_sim)
                    
                    stats['component']['total'] += 1
                    if comp_sim > neg_sim:
                        stats['component']['correct_higher'] += 1
                    else:
                        stats['component']['incorrect_higher'] += 1
        
        # Relation statistics
        for rel in sample.relations[:2]:
            subject = rel.get('subject', '')
            relation_type = rel.get('relation_type', '')
            obj = rel.get('object', '')
            
            if subject and relation_type and obj:
                rel_text = f"{subject} {relation_type} {obj}"
                rel_emb = extractor.encode_text(rel_text)
                rel_sim = extractor.compute_similarity(image_emb, rel_emb)
                similarities['relation_correct'].append(rel_sim)
                
                # Swapped
                swapped = f"{obj} {relation_type} {subject}"
                swapped_emb = extractor.encode_text(swapped)
                swapped_sim = extractor.compute_similarity(image_emb, swapped_emb)
                similarities['relation_swapped'].append(swapped_sim)
                
                stats['relation']['total'] += 1
                if rel_sim > swapped_sim:
                    stats['relation']['correct_higher'] += 1
                else:
                    stats['relation']['incorrect_higher'] += 1
        
        # Binding statistics
        for binding in sample.binding_negatives[:2]:
            comp1 = binding.get('component_1', '')
            bind_neg1 = binding.get('binding_neg_1', '')
            
            if comp1 and bind_neg1:
                emb1 = extractor.encode_text(comp1)
                sim1 = extractor.compute_similarity(image_emb, emb1)
                similarities['binding_correct'].append(sim1)
                
                neg_emb1 = extractor.encode_text(bind_neg1)
                neg_sim1 = extractor.compute_similarity(image_emb, neg_emb1)
                similarities['binding_swapped'].append(neg_sim1)
                
                stats['binding']['total'] += 1
                if sim1 > neg_sim1:
                    stats['binding']['correct_higher'] += 1
                else:
                    stats['binding']['incorrect_higher'] += 1
        
        # Half-truth statistics
        if len(sample.components) >= 2:
            comp_a = sample.components[0]
            comp_b = sample.components[1]
            negs_b = sample.component_negatives.get(comp_b, [])
            
            if negs_b:
                neg_b = negs_b[0].get('negative', '') if isinstance(negs_b[0], dict) else negs_b[0]
                if neg_b:
                    short_correct = comp_a
                    long_incorrect = f"{comp_a} and {neg_b}"
                    
                    short_emb = extractor.encode_text(short_correct)
                    short_sim = extractor.compute_similarity(image_emb, short_emb)
                    
                    long_emb = extractor.encode_text(long_incorrect)
                    long_sim = extractor.compute_similarity(image_emb, long_emb)
                    
                    stats['half_truth']['total'] += 1
                    if short_sim > long_sim:
                        stats['half_truth']['short_correct_wins'] += 1
                    else:
                        stats['half_truth']['long_incorrect_wins'] += 1
    
    # Compute summary
    summary = {
        'stats': stats,
        'similarity_means': {k: float(np.mean(v)) if v else 0 for k, v in similarities.items()},
        'similarity_stds': {k: float(np.std(v)) if v else 0 for k, v in similarities.items()},
        'accuracy': {
            'component': stats['component']['correct_higher'] / max(1, stats['component']['total']),
            'relation': stats['relation']['correct_higher'] / max(1, stats['relation']['total']),
            'binding': stats['binding']['correct_higher'] / max(1, stats['binding']['total']),
            'half_truth_resistance': stats['half_truth']['short_correct_wins'] / max(1, stats['half_truth']['total']),
        }
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ATTENTION & SIMILARITY STATISTICS")
    print("=" * 60)
    print(f"\nComponent Accuracy: {summary['accuracy']['component']:.1%}")
    print(f"  Positive mean sim: {summary['similarity_means']['component_positive']:.3f}")
    print(f"  Negative mean sim: {summary['similarity_means']['component_negative']:.3f}")
    print(f"\nRelation Accuracy (correct > swapped): {summary['accuracy']['relation']:.1%}")
    print(f"  Correct mean sim: {summary['similarity_means']['relation_correct']:.3f}")
    print(f"  Swapped mean sim: {summary['similarity_means']['relation_swapped']:.3f}")
    print(f"\nBinding Accuracy (correct > swapped): {summary['accuracy']['binding']:.1%}")
    print(f"  Correct mean sim: {summary['similarity_means']['binding_correct']:.3f}")
    print(f"  Swapped mean sim: {summary['similarity_means']['binding_swapped']:.3f}")
    print(f"\nHalf-Truth Resistance: {summary['accuracy']['half_truth_resistance']:.1%}")
    print(f"  (Short correct wins over long incorrect)")
    print("=" * 60)
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Attention Visualization for CLIP")
    
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
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="attention_viz_results")
    
    # Visualization options
    parser.add_argument("--skip_components", action="store_true")
    parser.add_argument("--skip_relations", action="store_true")
    parser.add_argument("--skip_bindings", action="store_true")
    parser.add_argument("--skip_half_truth", action="store_true")
    parser.add_argument("--compute_stats", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "components").mkdir(exist_ok=True)
    (output_dir / "relations").mkdir(exist_ok=True)
    (output_dir / "bindings").mkdir(exist_ok=True)
    (output_dir / "half_truth").mkdir(exist_ok=True)
    
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
    extractor = AttentionExtractor(
        model_name=args.model_name,
        checkpoint_type=args.checkpoint_type,
        checkpoint_path=args.checkpoint_path,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
    )
    
    # Generate visualizations
    for sample in tqdm(samples, desc="Generating attention visualizations"):
        if not args.skip_components:
            visualize_component_attention(extractor, sample, output_dir / "components")
        
        if not args.skip_relations:
            visualize_relation_attention(extractor, sample, output_dir / "relations")
        
        if not args.skip_bindings:
            visualize_binding_attention(extractor, sample, output_dir / "bindings")
        
        if not args.skip_half_truth:
            visualize_half_truth_attention(extractor, sample, output_dir / "half_truth")
    
    # Compute statistics
    if args.compute_stats:
        compute_attention_statistics(
            extractor,
            samples,
            str(output_dir / "attention_statistics.json"),
        )
    
    logger.info(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
