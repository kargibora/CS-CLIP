#!/usr/bin/env python3
"""
CLIP Behavior Analysis - Understanding model attention with positive/negative components.

This script provides multiple visualization methods to understand:
1. Where CLIP "looks" for different text queries
2. How behavior changes between positive and negative components
3. How fine-tuned models differ from baseline CLIP

Methods implemented:
- Occlusion sensitivity (perturbation-based)
- RISE (randomized input sampling)
- Attention rollout
- Similarity landscapes
- Component contribution analysis
- Deletion/insertion curves

Usage:
    python experiments/clip_behavior_analysis.py \
        --json_folder swap_pos_json/coco_val \
        --image_root . \
        --output_dir behavior_analysis \
        --num_samples 30 \
        --compare_checkpoints baseline /path/to/checkpoint.pt \
        --checkpoint_names "CLIP" "CS-CLIP" \
        --methods occlusion landscape tokens margins
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import clip
except ImportError:
    print("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentPair:
    """A positive component with its negative variants."""
    positive: str
    negatives: List[str]
    swap_type: str  # "attribute", "object", "relation", "binding"
    
@dataclass 
class SampleData:
    """Complete sample with image and component pairs."""
    image_path: str
    full_caption: str
    components: List[ComponentPair]
    sample_id: str


# =============================================================================
# CLIP Behavior Analyzer
# =============================================================================

class CLIPBehaviorAnalyzer:
    """Comprehensive CLIP behavior analysis with multiple visualization methods."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "external",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load model
        logger.info(f"Loading model {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, checkpoint_type)
        
        self.model.eval()
        
        # Get model info
        self.patch_size = self.model.visual.conv1.kernel_size[0]
        self.input_size = self.model.visual.input_resolution
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        
        logger.info(f"Model loaded. Patch size: {self.patch_size}, Grid: {self.grid_size}x{self.grid_size}")
    
    def _load_checkpoint(self, path: str, checkpoint_type: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Try to load
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Remove unexpected keys and try again
            model_keys = set(self.model.state_dict().keys())
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            self.model.load_state_dict(state_dict, strict=False)
            logger.warning("Loaded checkpoint with strict=False")
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to normalized feature vector."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = F.normalize(features, dim=-1)
        return features
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to normalized feature vector."""
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features = F.normalize(features, dim=-1)
        return features
    
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute image-text similarity."""
        img_feat = self.encode_image(image)
        txt_feat = self.encode_text(text)
        return (img_feat @ txt_feat.T).item()
    
    # =========================================================================
    # Method 1: Occlusion Sensitivity
    # =========================================================================
    
    def compute_occlusion_sensitivity(
        self,
        image: Image.Image,
        text: str,
        patch_size: int = 32,
        stride: int = 16,
        occlusion_value: str = "mean",  # "mean", "zero", "blur"
    ) -> Tuple[np.ndarray, float]:
        """
        Compute occlusion sensitivity map.
        
        For each patch location, occlude that region and measure
        how much the similarity drops. High drop = important region.
        """
        # Get baseline similarity
        baseline_sim = self.compute_similarity(image, text)
        
        # Convert to numpy for manipulation
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Compute occlusion value
        if occlusion_value == "mean":
            occlude_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
        elif occlusion_value == "zero":
            occlude_val = np.zeros(3, dtype=np.uint8)
        else:  # blur - we'll use mean for simplicity
            occlude_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
        
        # Compute sensitivity at each location
        n_rows = (h - patch_size) // stride + 1
        n_cols = (w - patch_size) // stride + 1
        sensitivity_map = np.zeros((n_rows, n_cols))
        
        for i in range(n_rows):
            for j in range(n_cols):
                # Create occluded image
                occluded = img_array.copy()
                y_start = i * stride
                x_start = j * stride
                occluded[y_start:y_start+patch_size, x_start:x_start+patch_size] = occlude_val
                
                # Compute similarity with occluded image
                occluded_img = Image.fromarray(occluded)
                occluded_sim = self.compute_similarity(occluded_img, text)
                
                # Sensitivity = drop in similarity (higher = more important)
                sensitivity_map[i, j] = baseline_sim - occluded_sim
        
        # Normalize
        if sensitivity_map.max() > sensitivity_map.min():
            sensitivity_map = (sensitivity_map - sensitivity_map.min()) / \
                             (sensitivity_map.max() - sensitivity_map.min())
        
        return sensitivity_map, baseline_sim
    
    # =========================================================================
    # Method 2: RISE (Randomized Input Sampling for Explanation)
    # =========================================================================
    
    def compute_rise_saliency(
        self,
        image: Image.Image,
        text: str,
        n_masks: int = 1000,
        mask_prob: float = 0.5,
        mask_resolution: int = 7,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute RISE saliency map.
        
        Sample random binary masks, weight each mask by the similarity
        score it produces, and average to get importance.
        """
        baseline_sim = self.compute_similarity(image, text)
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Generate random masks at low resolution, then upsample
        saliency = np.zeros((h, w))
        total_weight = np.zeros((h, w))
        
        for _ in tqdm(range(n_masks), desc="RISE sampling", leave=False):
            # Generate low-res random mask
            small_mask = np.random.binomial(1, mask_prob, (mask_resolution, mask_resolution))
            
            # Upsample to image size with some randomness in position
            # Add padding for random shift
            pad = mask_resolution
            padded = np.zeros((mask_resolution + 2*pad, mask_resolution + 2*pad))
            padded[pad:pad+mask_resolution, pad:pad+mask_resolution] = small_mask
            
            # Random shift
            shift_y = np.random.randint(0, 2*pad)
            shift_x = np.random.randint(0, 2*pad)
            shifted = padded[shift_y:shift_y+mask_resolution, shift_x:shift_x+mask_resolution]
            
            # Upsample to image size
            mask = np.array(Image.fromarray(shifted.astype(np.float32)).resize((w, h), Image.BILINEAR))
            
            # Apply mask to image
            masked_img = (img_array * mask[:, :, np.newaxis]).astype(np.uint8)
            masked_pil = Image.fromarray(masked_img)
            
            # Get similarity
            sim = self.compute_similarity(masked_pil, text)
            
            # Weight the mask by similarity
            saliency += mask * sim
            total_weight += mask
        
        # Normalize
        saliency = saliency / (total_weight + 1e-8)
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency, baseline_sim
    
    # =========================================================================
    # Method 3: Deletion/Insertion Curves
    # =========================================================================
    
    def compute_deletion_insertion_curves(
        self,
        image: Image.Image,
        text: str,
        saliency_map: np.ndarray,
        n_steps: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Compute deletion and insertion curves.
        
        Deletion: Start with full image, progressively remove most important patches
        Insertion: Start with blank, progressively add most important patches
        
        Good saliency = deletion curve drops fast, insertion curve rises fast
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Resize saliency to image size
        saliency_resized = np.array(
            Image.fromarray(saliency_map.astype(np.float32)).resize((w, h), Image.BILINEAR)
        )
        
        # Get pixel importance order
        importance_order = np.argsort(saliency_resized.flatten())[::-1]  # Most to least important
        
        n_pixels = h * w
        step_size = n_pixels // n_steps
        
        deletion_scores = []
        insertion_scores = []
        
        # Compute mean value for occlusion
        mean_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
        
        for step in range(n_steps + 1):
            n_modified = step * step_size
            
            # Deletion: remove top-n important pixels
            deleted = img_array.copy().reshape(-1, 3)
            if n_modified > 0:
                deleted[importance_order[:n_modified]] = mean_val
            deleted = deleted.reshape(h, w, 3)
            del_sim = self.compute_similarity(Image.fromarray(deleted), text)
            deletion_scores.append(del_sim)
            
            # Insertion: show only top-n important pixels
            inserted = np.full_like(img_array, mean_val)
            inserted = inserted.reshape(-1, 3)
            if n_modified > 0:
                flat_img = img_array.reshape(-1, 3)
                inserted[importance_order[:n_modified]] = flat_img[importance_order[:n_modified]]
            inserted = inserted.reshape(h, w, 3)
            ins_sim = self.compute_similarity(Image.fromarray(inserted), text)
            insertion_scores.append(ins_sim)
        
        # Compute AUC (Area Under Curve)
        x = np.linspace(0, 1, n_steps + 1)
        deletion_auc = np.trapz(deletion_scores, x)
        insertion_auc = np.trapz(insertion_scores, x)
        
        return {
            'deletion_scores': np.array(deletion_scores),
            'insertion_scores': np.array(insertion_scores),
            'deletion_auc': deletion_auc,
            'insertion_auc': insertion_auc,
            'x': x,
        }
    
    # =========================================================================
    # Method 4: Similarity Landscape
    # =========================================================================
    
    def compute_similarity_landscape(
        self,
        image: Image.Image,
        text_positive: str,
        text_negative: str,
        n_interpolations: int = 11,
    ) -> Dict[str, Any]:
        """
        Compute similarity as we interpolate between positive and negative text embeddings.
        
        This shows how "confident" the model is and where the decision boundary lies.
        """
        img_feat = self.encode_image(image)
        pos_feat = self.encode_text(text_positive)
        neg_feat = self.encode_text(text_negative)
        
        # Interpolate between negative and positive
        alphas = np.linspace(0, 1, n_interpolations)
        similarities = []
        
        for alpha in alphas:
            # Linear interpolation in embedding space
            interp_feat = (1 - alpha) * neg_feat + alpha * pos_feat
            interp_feat = F.normalize(interp_feat, dim=-1)
            sim = (img_feat @ interp_feat.T).item()
            similarities.append(sim)
        
        # Find crossover point (where sim to interpolated equals some threshold)
        pos_sim = self.compute_similarity(image, text_positive)
        neg_sim = self.compute_similarity(image, text_negative)
        margin = pos_sim - neg_sim
        
        return {
            'alphas': alphas,
            'similarities': np.array(similarities),
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
            'margin': margin,
            'text_positive': text_positive,
            'text_negative': text_negative,
        }
    
    # =========================================================================
    # Method 5: Token Contribution Analysis
    # =========================================================================
    
    def compute_token_contributions(
        self,
        image: Image.Image,
        text: str,
    ) -> Dict[str, Any]:
        """
        Analyze how each token in the text contributes to the similarity.
        
        Uses leave-one-out: compute similarity with each token masked.
        """
        baseline_sim = self.compute_similarity(image, text)
        
        # Split text into words (approximate since CLIP uses BPE)
        words = text.split()
        
        contributions = []
        
        for i, word in enumerate(words):
            # Create text with word removed
            masked_words = words[:i] + words[i+1:]
            masked_text = " ".join(masked_words)
            
            if masked_text.strip():
                masked_sim = self.compute_similarity(image, masked_text)
                contribution = baseline_sim - masked_sim  # How much removing this word hurts
            else:
                contribution = baseline_sim  # Removing everything
            
            contributions.append({
                'word': word,
                'contribution': contribution,
                'position': i,
            })
        
        return {
            'baseline_sim': baseline_sim,
            'contributions': contributions,
            'text': text,
        }
    
    # =========================================================================
    # Method 6: Positive vs Negative Saliency Comparison
    # =========================================================================
    
    def compute_pos_neg_comparison(
        self,
        image: Image.Image,
        text_positive: str,
        text_negative: str,
        method: str = "occlusion",  # "occlusion", "rise", "gradient"
    ) -> Dict[str, Any]:
        """
        Compare saliency maps for positive vs negative text.
        
        Returns individual maps and difference map.
        """
        if method == "occlusion":
            smap_pos, sim_pos = self.compute_occlusion_sensitivity(image, text_positive)
            smap_neg, sim_neg = self.compute_occlusion_sensitivity(image, text_negative)
        elif method == "rise":
            smap_pos, sim_pos = self.compute_rise_saliency(image, text_positive, n_masks=500)
            smap_neg, sim_neg = self.compute_rise_saliency(image, text_negative, n_masks=500)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Resize to same size for comparison
        target_size = max(smap_pos.shape[0], smap_neg.shape[0])
        smap_pos_resized = np.array(
            Image.fromarray(smap_pos.astype(np.float32)).resize(
                (target_size, target_size), Image.BILINEAR
            )
        )
        smap_neg_resized = np.array(
            Image.fromarray(smap_neg.astype(np.float32)).resize(
                (target_size, target_size), Image.BILINEAR
            )
        )
        
        # Difference: where does positive focus more than negative?
        diff_map = smap_pos_resized - smap_neg_resized
        
        return {
            'smap_positive': smap_pos_resized,
            'smap_negative': smap_neg_resized,
            'diff_map': diff_map,
            'sim_positive': sim_pos,
            'sim_negative': sim_neg,
            'margin': sim_pos - sim_neg,
            'text_positive': text_positive,
            'text_negative': text_negative,
        }


# =============================================================================
# Multi-Model Comparison
# =============================================================================

class MultiModelAnalyzer:
    """Compare behavior across multiple models."""
    
    def __init__(self, models: Dict[str, CLIPBehaviorAnalyzer]):
        self.models = models
    
    def compare_similarities(
        self,
        image: Image.Image,
        texts: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare similarities across models for multiple texts."""
        results = {}
        for model_name, analyzer in self.models.items():
            results[model_name] = {}
            for text in texts:
                results[model_name][text] = analyzer.compute_similarity(image, text)
        return results
    
    def compare_margins(
        self,
        image: Image.Image,
        text_positive: str,
        text_negative: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compare positive-negative margins across models."""
        results = {}
        for model_name, analyzer in self.models.items():
            sim_pos = analyzer.compute_similarity(image, text_positive)
            sim_neg = analyzer.compute_similarity(image, text_negative)
            results[model_name] = {
                'positive': sim_pos,
                'negative': sim_neg,
                'margin': sim_pos - sim_neg,
                'correct': sim_pos > sim_neg,
            }
        return results
    
    def compare_saliency(
        self,
        image: Image.Image,
        text: str,
        method: str = "occlusion",
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Compare saliency maps across models."""
        results = {}
        for model_name, analyzer in self.models.items():
            if method == "occlusion":
                smap, sim = analyzer.compute_occlusion_sensitivity(image, text)
            elif method == "rise":
                smap, sim = analyzer.compute_rise_saliency(image, text, n_masks=300)
            else:
                raise ValueError(f"Unknown method: {method}")
            results[model_name] = (smap, sim)
        return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_occlusion_comparison(
    image: Image.Image,
    results: Dict[str, Dict[str, Any]],  # model_name -> pos_neg_comparison result
    save_path: str,
    sample_id: str = "",
):
    """
    Plot occlusion sensitivity comparison across models.
    
    Layout:
    - Row 1: Original | Model1 Pos | Model1 Neg | Model1 Diff | ...
    - Colors: Positive focus (green), Negative focus (red)
    """
    n_models = len(results)
    fig, axes = plt.subplots(2, 1 + n_models * 3, figsize=(4 * (1 + n_models * 3), 8))

    for row in range(2):
        # Original image
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original Image", fontsize=10)
        axes[row, 0].axis('off')
        
        col = 1
        for model_name, result in results.items():
            if row == 0:
                smap = result['smap_positive']
                text = result['text_positive'][:30] + "..."
                sim = result['sim_positive']
            else:
                smap = result['smap_negative']
                text = result['text_negative'][:30] + "..."
                sim = result['sim_negative']
            
            # Resize saliency to image size
            smap_resized = np.array(
                Image.fromarray(smap.astype(np.float32)).resize(
                    image.size, Image.BILINEAR
                )
            )
            
            # Overlay on image
            img_array = np.array(image) / 255.0
            heatmap = plt.cm.jet(smap_resized)[:, :, :3]
            overlay = 0.6 * img_array + 0.4 * heatmap
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"{model_name}\n{text}\nsim={sim:.3f}", fontsize=9)
            axes[row, col].axis('off')
            col += 1
        
        # Add difference maps for first row only
        if row == 0:
            for model_name, result in results.items():
                diff = result['diff_map']
                diff_resized = np.array(
                    Image.fromarray(diff.astype(np.float32)).resize(
                        image.size, Image.BILINEAR
                    )
                )
                
                # Use diverging colormap
                vmax = max(abs(diff_resized.min()), abs(diff_resized.max()))
                axes[row, col].imshow(diff_resized, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[row, col].set_title(f"{model_name}\nDiff (Pos - Neg)\nmargin={result['margin']:.3f}", fontsize=9)
                axes[row, col].axis('off')
                col += 1
    
    plt.suptitle(f"Occlusion Sensitivity: {sample_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_landscape_comparison(
    image: Image.Image,
    landscapes: Dict[str, Dict[str, Any]],  # model_name -> landscape result
    save_path: str,
    sample_id: str = "",
):
    """
    Plot similarity landscapes across models.
    
    Shows how similarity changes as we interpolate from negative to positive text.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Image
    axes[0].imshow(image)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis('off')
    
    # Right: Similarity curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(landscapes)))
    
    for (model_name, landscape), color in zip(landscapes.items(), colors):
        alphas = landscape['alphas']
        sims = landscape['similarities']
        
        axes[1].plot(alphas, sims, label=model_name, color=color, linewidth=2)
        
        # Mark endpoints
        axes[1].scatter([0], [landscape['neg_sim']], color=color, marker='x', s=100)
        axes[1].scatter([1], [landscape['pos_sim']], color=color, marker='o', s=100)
    
    axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Interpolation α (0=Negative, 1=Positive)", fontsize=11)
    axes[1].set_ylabel("Image-Text Similarity", fontsize=11)
    axes[1].legend(loc='best')
    axes[1].set_title("Similarity Landscape", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add text annotations
    first_landscape = list(landscapes.values())[0]
    pos_text = first_landscape['text_positive'][:40] + "..."
    neg_text = first_landscape['text_negative'][:40] + "..."
    
    axes[1].annotate(f"Neg: {neg_text}", xy=(0.02, 0.98), xycoords='axes fraction',
                     fontsize=8, va='top', color='red')
    axes[1].annotate(f"Pos: {pos_text}", xy=(0.02, 0.93), xycoords='axes fraction',
                     fontsize=8, va='top', color='green')
    
    plt.suptitle(f"Similarity Landscape: {sample_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_token_contributions_comparison(
    image: Image.Image,
    contributions: Dict[str, Dict[str, Any]],  # model_name -> token contribution result
    save_path: str,
    sample_id: str = "",
):
    """
    Plot token contributions across models as bar charts.
    """
    n_models = len(contributions)
    
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
    
    # Image
    axes[0].imshow(image)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis('off')
    
    for idx, (model_name, result) in enumerate(contributions.items()):
        ax = axes[idx + 1]
        
        words = [c['word'] for c in result['contributions']]
        contribs = [c['contribution'] for c in result['contributions']]
        
        colors = ['green' if c > 0 else 'red' for c in contribs]
        
        ax.barh(range(len(words)), contribs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel("Contribution to Similarity", fontsize=10)
        ax.set_title(f"{model_name}\nBaseline sim={result['baseline_sim']:.3f}", fontsize=11)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.invert_yaxis()  # Most important at top
    
    plt.suptitle(f"Token Contributions: {sample_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_deletion_insertion_curves(
    curves: Dict[str, Dict[str, Any]],  # model_name -> curves result
    save_path: str,
    sample_id: str = "",
):
    """
    Plot deletion and insertion curves for evaluating saliency quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    
    # Deletion curves
    for (model_name, result), color in zip(curves.items(), colors):
        x = result['x']
        axes[0].plot(x, result['deletion_scores'], label=f"{model_name} (AUC={result['deletion_auc']:.3f})",
                     color=color, linewidth=2)
    
    axes[0].set_xlabel("Fraction of Pixels Removed", fontsize=11)
    axes[0].set_ylabel("Similarity", fontsize=11)
    axes[0].set_title("Deletion Curve (Lower = Better Saliency)", fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Insertion curves
    for (model_name, result), color in zip(curves.items(), colors):
        x = result['x']
        axes[1].plot(x, result['insertion_scores'], label=f"{model_name} (AUC={result['insertion_auc']:.3f})",
                     color=color, linewidth=2)
    
    axes[1].set_xlabel("Fraction of Pixels Inserted", fontsize=11)
    axes[1].set_ylabel("Similarity", fontsize=11)
    axes[1].set_title("Insertion Curve (Higher = Better Saliency)", fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Saliency Quality Evaluation: {sample_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_margin_comparison(
    margins: List[Dict[str, Dict[str, float]]],  # List of margin results per sample
    model_names: List[str],
    save_path: str,
):
    """
    Plot margin distribution across samples for each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect margins per model
    model_margins = {name: [] for name in model_names}
    model_correct = {name: 0 for name in model_names}
    
    for margin_result in margins:
        for model_name in model_names:
            if model_name in margin_result:
                model_margins[model_name].append(margin_result[model_name]['margin'])
                if margin_result[model_name]['correct']:
                    model_correct[model_name] += 1
    
    # Box plot of margins
    data = [model_margins[name] for name in model_names]
    bp = axes[0].boxplot(data, labels=model_names, patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_ylabel("Margin (Positive - Negative)", fontsize=11)
    axes[0].set_title("Similarity Margin Distribution", fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy bar chart
    n_samples = len(margins)
    accuracies = [model_correct[name] / n_samples * 100 for name in model_names]
    
    bars = axes[1].bar(model_names, accuracies, color=colors, alpha=0.7)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("Correct Ranking (Pos > Neg)", fontsize=12)
    axes[1].set_ylim(0, 100)
    
    # Add percentage labels
    for bar, acc in zip(bars, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Data Loading
# =============================================================================

def load_samples(json_folder: str, image_root: str, num_samples: int = 50) -> List[SampleData]:
    """Load samples from JSON files."""
    samples = []
    json_folder = Path(json_folder)
    
    json_files = list(json_folder.glob("*.json"))
    random.shuffle(json_files)
    
    for json_file in json_files[:num_samples * 2]:  # Load extra in case some fail
        if len(samples) >= num_samples:
            break
            
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Get image path
            image_path = data.get('image', data.get('image_path', ''))
            if not image_path:
                continue
            
            full_path = os.path.join(image_root, image_path)
            if not os.path.exists(full_path):
                continue
            
            # Get components
            components = []
            for comp in data.get('component_captions', []):
                if comp.get('negatives'):
                    components.append(ComponentPair(
                        positive=comp['caption'],
                        negatives=[n['caption'] for n in comp['negatives'][:3]],
                        swap_type=comp['negatives'][0].get('swap_type', 'unknown') if comp['negatives'] else 'unknown',
                    ))
            
            if not components:
                continue
            
            samples.append(SampleData(
                image_path=full_path,
                full_caption=data.get('text', data.get('caption', '')),
                components=components,
                sample_id=json_file.stem,
            ))
            
        except Exception:
            continue
    
    return samples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CLIP Behavior Analysis")
    parser.add_argument("--json_folder", type=str, required=True, help="Folder with JSON annotations")
    parser.add_argument("--image_root", type=str, default=".", help="Root folder for images")
    parser.add_argument("--output_dir", type=str, default="behavior_analysis", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to analyze")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--compare_checkpoints", nargs="+", default=None,
                        help="Checkpoints to compare (use 'baseline' for pretrained CLIP)")
    parser.add_argument("--checkpoint_names", nargs="+", default=None,
                        help="Names for each checkpoint")
    parser.add_argument("--checkpoint_type", type=str, default="external",
                        help="Checkpoint type for loading")
    
    # Analysis methods
    parser.add_argument("--methods", nargs="+", 
                        default=["occlusion", "landscape", "tokens", "margins"],
                        choices=["occlusion", "rise", "landscape", "tokens", "deletion", "margins", "all"],
                        help="Analysis methods to run")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expand 'all' methods
    if "all" in args.methods:
        args.methods = ["occlusion", "landscape", "tokens", "margins"]
    
    # Load models
    models = {}
    
    if args.compare_checkpoints:
        for i, ckpt in enumerate(args.compare_checkpoints):
            name = args.checkpoint_names[i] if args.checkpoint_names and i < len(args.checkpoint_names) else f"Model_{i}"
            
            if ckpt.lower() == "baseline":
                logger.info(f"Loading baseline CLIP as '{name}'")
                models[name] = CLIPBehaviorAnalyzer(
                    model_name=args.model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            else:
                logger.info(f"Loading checkpoint '{ckpt}' as '{name}'")
                models[name] = CLIPBehaviorAnalyzer(
                    model_name=args.model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    checkpoint_path=ckpt,
                    checkpoint_type=args.checkpoint_type,
                )
    else:
        # Just baseline
        models["CLIP"] = CLIPBehaviorAnalyzer(
            model_name=args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    # Create multi-model analyzer
    multi_analyzer = MultiModelAnalyzer(models)
    
    # Load samples
    logger.info(f"Loading samples from {args.json_folder}")
    samples = load_samples(args.json_folder, args.image_root, args.num_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create subdirectories
    for method in args.methods:
        (output_dir / method).mkdir(exist_ok=True)
    
    # Collect statistics
    all_margins = []
    
    # Process samples
    for idx, sample in enumerate(tqdm(samples, desc="Analyzing samples")):
        try:
            image = Image.open(sample.image_path).convert("RGB")
            
            # Use first component for analysis
            if not sample.components:
                continue
            
            component = sample.components[0]
            text_pos = component.positive
            text_neg = component.negatives[0] if component.negatives else text_pos
            
            sample_id = f"{idx:03d}_{sample.sample_id}"
            
            # -----------------------------------------------------------------
            # Occlusion Sensitivity
            # -----------------------------------------------------------------
            if "occlusion" in args.methods:
                occlusion_results = {}
                for model_name, analyzer in models.items():
                    occlusion_results[model_name] = analyzer.compute_pos_neg_comparison(
                        image, text_pos, text_neg, method="occlusion"
                    )
                
                plot_occlusion_comparison(
                    image, occlusion_results,
                    str(output_dir / "occlusion" / f"occlusion_{sample_id}.png"),
                    sample_id
                )
            
            # -----------------------------------------------------------------
            # Similarity Landscape
            # -----------------------------------------------------------------
            if "landscape" in args.methods:
                landscapes = {}
                for model_name, analyzer in models.items():
                    landscapes[model_name] = analyzer.compute_similarity_landscape(
                        image, text_pos, text_neg
                    )
                
                plot_similarity_landscape_comparison(
                    image, landscapes,
                    str(output_dir / "landscape" / f"landscape_{sample_id}.png"),
                    sample_id
                )
            
            # -----------------------------------------------------------------
            # Token Contributions
            # -----------------------------------------------------------------
            if "tokens" in args.methods:
                contributions = {}
                for model_name, analyzer in models.items():
                    contributions[model_name] = analyzer.compute_token_contributions(
                        image, text_pos
                    )
                
                plot_token_contributions_comparison(
                    image, contributions,
                    str(output_dir / "tokens" / f"tokens_{sample_id}.png"),
                    sample_id
                )
            
            # -----------------------------------------------------------------
            # Margins (collect for summary)
            # -----------------------------------------------------------------
            if "margins" in args.methods:
                margin_result = multi_analyzer.compare_margins(image, text_pos, text_neg)
                all_margins.append(margin_result)
            
            # -----------------------------------------------------------------
            # Deletion/Insertion Curves (slow, do for fewer samples)
            # -----------------------------------------------------------------
            if "deletion" in args.methods and idx < 10:
                curves = {}
                for model_name, analyzer in models.items():
                    # First compute saliency
                    smap, _ = analyzer.compute_occlusion_sensitivity(image, text_pos)
                    curves[model_name] = analyzer.compute_deletion_insertion_curves(
                        image, text_pos, smap
                    )
                
                plot_deletion_insertion_curves(
                    curves,
                    str(output_dir / "deletion" / f"deletion_{sample_id}.png"),
                    sample_id
                )
        
        except Exception:
            logger.warning(f"Error processing sample {idx}")
            continue
    
    # -----------------------------------------------------------------
    # Summary plots
    # -----------------------------------------------------------------
    if "margins" in args.methods and all_margins:
        plot_margin_comparison(
            all_margins,
            list(models.keys()),
            str(output_dir / "margin_summary.png")
        )
        
        # Save statistics
        stats = {}
        for model_name in models.keys():
            margins = [m[model_name]['margin'] for m in all_margins if model_name in m]
            correct = sum(1 for m in all_margins if model_name in m and m[model_name]['correct'])
            stats[model_name] = {
                'mean_margin': float(np.mean(margins)),
                'std_margin': float(np.std(margins)),
                'accuracy': correct / len(all_margins) * 100,
                'n_samples': len(all_margins),
            }
        
        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\nSummary Statistics:")
        for model_name, s in stats.items():
            logger.info(f"  {model_name}: Acc={s['accuracy']:.1f}%, Margin={s['mean_margin']:.4f}±{s['std_margin']:.4f}")
    
    logger.info(f"\n✅ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
