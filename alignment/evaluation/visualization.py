"""Visualization utilities for evaluation debugging."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import torch


class SimilarityVisualizer:
    """
    Creates debugging visualizations for component-image similarities.
    
    Tracks the same samples across evaluation runs to show how similarities evolve.
    """
    
    def __init__(
        self,
        num_samples: int = 5,
        seed: int = 42,
        output_dir: str = "./debug_visualizations",
        denormalize_fn: Optional[callable] = None
    ):
        """
        Args:
            num_samples: Number of samples to visualize (5-10 recommended)
            seed: Random seed for sample selection (ensures consistency)
            output_dir: Directory to save visualizations
            denormalize_fn: Function to denormalize images for display
        """
        self.num_samples = num_samples
        self.seed = seed
        self.output_dir = output_dir
        self.denormalize_fn = denormalize_fn
        self.selected_indices = None  # Will be set on first call
        
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"SimilarityVisualizer initialized: {num_samples} samples, seed={seed}, output={output_dir}")
    
    def select_samples(self, dataset_size: int) -> List[int]:
        """
        Select fixed sample indices for consistent visualization.
        
        Args:
            dataset_size: Total size of validation dataset
            
        Returns:
            List of sample indices
        """
        if self.selected_indices is not None:
            return self.selected_indices
        
        # Use fixed seed for reproducibility
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(dataset_size, size=min(self.num_samples, dataset_size), replace=False)
        self.selected_indices = sorted(indices.tolist())
        
        logging.info(f"Selected visualization samples: {self.selected_indices}")
        return self.selected_indices
    
    def denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert normalized tensor to displayable numpy array.
        
        Args:
            img_tensor: [C, H, W] tensor
            
        Returns:
            [H, W, C] numpy array in [0, 255]
        """
        if self.denormalize_fn is not None:
            img = self.denormalize_fn(img_tensor)
        else:
            # Default: assume ImageNet normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            img = img_tensor.cpu() * std + mean
        
        # Clip to [0, 1] and convert to numpy
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def compute_similarities(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_emb: [D] image embedding
            text_emb: [N, D] text embeddings (or [D] for single text)
            
        Returns:
            [N] similarity scores (or scalar for single text)
        """
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        
        # Ensure normalized
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        
        # Compute cosine similarity
        similarities = (image_emb @ text_emb.T).squeeze()
        
        return similarities
    
    def create_visualization(
        self,
        sample_idx: int,
        image: torch.Tensor,
        full_caption: str,
        component_captions: List[str],
        neg_caption: str,
        image_emb: torch.Tensor,
        full_caption_emb: torch.Tensor,
        component_embs: torch.Tensor,  # [N, D]
        neg_caption_emb: torch.Tensor,
        step: int,
        epoch: Optional[int] = None
    ) -> plt.Figure:
        """
        Create visualization for a single sample.
        
        Args:
            sample_idx: Index of the sample
            image: [C, H, W] image tensor
            full_caption: Full caption text
            component_captions: List of component caption texts
            neg_caption: Negative caption text
            image_emb: [D] image embedding
            full_caption_emb: [D] full caption embedding
            component_embs: [N, D] component embeddings
            neg_caption_emb: [D] negative caption embedding
            step: Training step
            epoch: Training epoch (optional)
            
        Returns:
            matplotlib Figure
        """
        # Compute similarities
        sim_full = self.compute_similarities(image_emb, full_caption_emb).item()
        sim_comps = self.compute_similarities(image_emb, component_embs).cpu().numpy()
        sim_neg = self.compute_similarities(image_emb, neg_caption_emb).item()
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Title
        title = f"Sample {sample_idx}"
        if epoch is not None:
            title += f" | Epoch {epoch}"
        title += f" | Step {step}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # === Top Left: Image ===
        ax_img = fig.add_subplot(gs[0, 0])
        img_display = self.denormalize_image(image)
        ax_img.imshow(img_display)
        ax_img.axis('off')
        ax_img.set_title("Image", fontsize=12, fontweight='bold')
        
        # === Top Right: Similarity Bar Chart ===
        ax_sim = fig.add_subplot(gs[0, 1])
        
        # Prepare data
        labels = ['Full Caption'] + [f'Comp {i+1}' for i in range(len(sim_comps))] + ['Negative']
        similarities = [sim_full] + sim_comps.tolist() + [sim_neg]
        colors = ['#2ecc71'] + ['#3498db'] * len(sim_comps) + ['#e74c3c']
        
        y_pos = np.arange(len(labels))
        bars = ax_sim.barh(y_pos, similarities, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            width = bar.get_width()
            ax_sim.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{sim:.3f}', va='center', fontsize=9)
        
        ax_sim.set_yticks(y_pos)
        ax_sim.set_yticklabels(labels, fontsize=9)
        ax_sim.set_xlabel('Cosine Similarity', fontsize=10)
        ax_sim.set_title('Image-Text Similarities', fontsize=12, fontweight='bold')
        ax_sim.set_xlim(-1.05, 1.15)
        ax_sim.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
        ax_sim.grid(axis='x', alpha=0.3)
        
        # === Bottom Left: Full Caption ===
        ax_full = fig.add_subplot(gs[1, :])
        ax_full.axis('off')
        
        full_text = f"FULL CAPTION (sim={sim_full:.3f}):\n{full_caption}"
        ax_full.text(0.02, 0.5, full_text, fontsize=10, 
                    verticalalignment='center', wrap=True,
                    bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.5))
        
        # === Bottom Right: Component Captions ===
        ax_comps = fig.add_subplot(gs[2, :])
        ax_comps.axis('off')
        
        comp_text = "COMPONENT CAPTIONS:\n"
        for i, (comp_cap, sim) in enumerate(zip(component_captions, sim_comps)):
            comp_text += f"  [{i+1}] (sim={sim:.3f}): {comp_cap}\n"
        
        comp_text += f"\nNEGATIVE CAPTION (sim={sim_neg:.3f}):\n  {neg_caption}"
        
        ax_comps.text(0.02, 0.98, comp_text, fontsize=9,
                     verticalalignment='top', wrap=True,
                     bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.5))
        
        return fig
    
    def visualize_batch(
        self,
        images: torch.Tensor,
        full_captions: List[str],
        component_captions_batch: List[List[str]],
        neg_captions: List[str],
        image_embs: torch.Tensor,
        full_caption_embs: torch.Tensor,
        component_embs_batch: torch.Tensor,  # [B, N, D]
        neg_caption_embs: torch.Tensor,
        step: int,
        epoch: Optional[int] = None,
        sample_indices: Optional[List[int]] = None
    ) -> List[str]:
        """
        Create visualizations for a batch of samples.
        
        Args:
            images: [B, C, H, W] image tensors
            full_captions: List of B full caption texts
            component_captions_batch: List of B lists of component captions
            neg_captions: List of B negative caption texts
            image_embs: [B, D] image embeddings
            full_caption_embs: [B, D] full caption embeddings
            component_embs_batch: [B, N, D] component embeddings
            neg_caption_embs: [B, D] negative caption embeddings
            step: Training step
            epoch: Training epoch (optional)
            sample_indices: List of sample indices in dataset (for tracking)
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for i in range(images.size(0)):
            # Get sample index (for filename)
            sample_idx = sample_indices[i] if sample_indices is not None else i
            
            # Create visualization
            fig = self.create_visualization(
                sample_idx=sample_idx,
                image=images[i],
                full_caption=full_captions[i],
                component_captions=component_captions_batch[i],
                neg_caption=neg_captions[i],
                image_emb=image_embs[i],
                full_caption_emb=full_caption_embs[i],
                component_embs=component_embs_batch[i],
                neg_caption_emb=neg_caption_embs[i],
                step=step,
                epoch=epoch
            )
            
            # Save figure
            epoch_str = f"epoch{epoch:03d}_" if epoch is not None else ""
            filename = f"sample{sample_idx:04d}_{epoch_str}step{step:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            saved_paths.append(filepath)
            
        logging.info(f"Saved {len(saved_paths)} visualizations to {self.output_dir}")
        return saved_paths
    
    def create_evolution_plot(
        self,
        sample_idx: int,
        similarity_history: Dict[str, List[Tuple[int, float]]]
    ) -> plt.Figure:
        """
        Create a plot showing how similarities evolve over training.
        
        Args:
            sample_idx: Sample index
            similarity_history: Dict mapping caption type to list of (step, similarity) tuples
                Example: {
                    'full': [(0, 0.5), (100, 0.6), ...],
                    'comp_0': [(0, 0.3), (100, 0.4), ...],
                    'negative': [(0, -0.1), (100, -0.2), ...]
                }
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {
            'full': '#2ecc71',
            'negative': '#e74c3c'
        }
        
        # Plot each caption type
        for caption_type, history in similarity_history.items():
            if not history:
                continue
            
            steps, sims = zip(*history)
            
            if caption_type == 'full':
                label = 'Full Caption'
                color = colors['full']
                linewidth = 2
                marker = 'o'
            elif caption_type == 'negative':
                label = 'Negative'
                color = colors['negative']
                linewidth = 2
                marker = 's'
            elif caption_type.startswith('comp_'):
                comp_idx = int(caption_type.split('_')[1])
                label = f'Component {comp_idx + 1}'
                color = f'C{comp_idx}'
                linewidth = 1.5
                marker = '^'
            else:
                label = caption_type
                color = None
                linewidth = 1
                marker = 'x'
            
            ax.plot(steps, sims, label=label, color=color, 
                   linewidth=linewidth, marker=marker, markersize=5, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(f'Similarity Evolution - Sample {sample_idx}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        return fig


def collect_visualization_batch(
    batch: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    selected_indices: List[int],
    batch_offset: int
) -> Optional[Dict[str, Any]]:
    """
    Extract samples from batch for visualization.
    
    Args:
        batch: Original batch dict from dataloader
        outputs: Model outputs with embeddings
        selected_indices: Global indices of samples to visualize
        batch_offset: Offset of current batch in dataset
        
    Returns:
        Dict with selected samples or None if no overlap
    """
    batch_size = outputs['image_embeddings'].size(0)
    batch_range = range(batch_offset, batch_offset + batch_size)
    
    # Find overlap between selected indices and current batch
    local_indices = []
    global_indices = []
    
    for idx in selected_indices:
        if idx in batch_range:
            local_idx = idx - batch_offset
            local_indices.append(local_idx)
            global_indices.append(idx)
    
    if not local_indices:
        return None
    
    # Extract data
    vis_data = {
        'images': batch['images'][local_indices],
        'full_captions': [batch['captions'][i] for i in local_indices],
        'neg_captions': [batch['neg_captions'][i] for i in local_indices],
        'image_embs': outputs['image_embeddings'][local_indices],
        'full_caption_embs': outputs['pos_text_embeddings'][local_indices, 0],  # First in sequence
        'neg_caption_embs': outputs['neg_text_embeddings'][local_indices],
        'global_indices': global_indices
    }
    
    # Handle component captions if present
    if 'component_captions' in batch:
        # component_captions is List[List[str]], shape [B][N]
        vis_data['component_captions'] = [batch['component_captions'][i] for i in local_indices]
        
        # Component embeddings: [B, N, D]
        if outputs['pos_text_embeddings'].size(1) > 1:
            vis_data['component_embs'] = outputs['pos_text_embeddings'][local_indices, 1:]  # Skip full caption
        else:
            # No components in embeddings
            vis_data['component_embs'] = None
            vis_data['component_captions'] = [[] for _ in local_indices]
    else:
        vis_data['component_captions'] = [[] for _ in local_indices]
        vis_data['component_embs'] = None
    
    return vis_data
