"""
Training Sample Logger for Tracking Model Learning Progression

This module provides functionality to track how a model learns on specific samples
throughout training. It logs sample similarities to wandb with visualizations,
allowing you to monitor:
  - How image-caption similarities evolve over epochs
  - How the model improves on full captions vs components
  - Comparison between base CLIP and current model

Key Features:
  - Fixed sample set (same samples every epoch) for consistent tracking
  - Precomputed base CLIP similarities (computed once)
  - Per-epoch visualization panels with images and similarity scores
  - Wandb integration with both images and metrics
"""

import logging
import torch
import numpy as np
import wandb
import clip
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import io
import textwrap


class TrainingSampleLogger:
    """
    Logger for tracking model learning on fixed samples during training.
    
    This class:
      1. Selects a fixed set of samples from the dataset
      2. Precomputes base CLIP similarities (once at initialization)
      3. Computes current model similarities each epoch
      4. Creates visualization panels showing progress
      5. Logs to wandb with images and metrics
    """
    
    def __init__(
        self,
        dataset,
        base_clip_model,
        preprocess,
        device: torch.device,
        num_samples: int = 10,
        max_components: int = 5,
        seed: int = 42,
        is_ft: bool = False
    ):
        """
        Initialize the training sample logger.
        
        Args:
            dataset: LAION dataset with samples
            base_clip_model: Base CLIP model for comparison
            preprocess: CLIP preprocessing function
            device: Device for computation
            num_samples: Number of samples to track (default: 10)
            max_components: Maximum number of components to visualize (default: 5)
            seed: Random seed for sample selection
            is_ft: Whether fine-tuning mode is active
        """
        self.device = device
        self.num_samples = num_samples
        self.max_components = max_components
        self.seed = seed
        self.is_ft = is_ft
        self.preprocess = preprocess
        self.base_clip_model = base_clip_model
        
        # Track whether images have been logged
        self.images_logged = False
        
        # Select fixed samples
        np.random.seed(seed)
        dataset_size = len(dataset)
        self.sample_indices = np.random.choice(
            dataset_size, 
            size=min(num_samples, dataset_size), 
            replace=False
        ).tolist()
        
        logging.info(f"TrainingSampleLogger: Selected {len(self.sample_indices)} fixed samples for tracking")
        
        # Extract sample data and precompute base CLIP similarities
        self.samples = []
        self._extract_and_compute_base_similarities(dataset)
        
        logging.info(f"TrainingSampleLogger: Initialized with {len(self.samples)} valid samples")
    
    def _extract_and_compute_base_similarities(self, dataset):
        """Extract samples and precompute base CLIP similarities."""
        logging.info(f"Extracting {len(self.sample_indices)} samples from dataset...")
        for idx in self.sample_indices:
            try:
                sample = dataset[idx]
                logging.info(f"Sample {idx} keys: {sample.keys() if isinstance(sample, dict) else type(sample)}")
                
                # Handle multi-caption mode
                if 'image_options' in sample:
                    image_tensor = sample['image_options']
                    captions = sample.get('caption_options', [])
                    num_positives = sample.get('num_positives', 1)
                    logging.info(f"Sample {idx}: Found image_options with {len(captions)} captions")
                    
                    # Extract negative captions
                    # Format: [full_caption, component_1, component_2, ..., component_N, negative_1, negative_2, ..., negative_M]
                    # num_positives includes full caption + components (so num_positives=1 means just full, no components)
                    # Everything after num_positives captions are negatives
                    negative_captions = captions[num_positives:] if len(captions) > num_positives else []
                    logging.info(f"Sample {idx}: num_positives={num_positives}, extracted {len(negative_captions)} negatives from captions")
                    
                elif 'image' in sample:
                    # Fallback to single-caption mode
                    image_tensor = sample['image']
                    captions = [sample['caption']]
                    num_positives = 1
                    negative_captions = []
                    logging.info(f"Sample {idx}: Found single image with caption")
                else:
                    logging.error(f"Sample {idx}: No image found! Keys: {sample.keys()}")
                    continue
                
                # Denormalize image for saving
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                image_denorm = image_tensor * std + mean
                image_denorm = torch.clamp(image_denorm, 0, 1)
                
                # Convert to PIL
                image_pil = Image.fromarray(
                    (image_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                
                # Parse captions
                full_caption = captions[0] if len(captions) > 0 else "No caption"
                component_captions = captions[1:num_positives] if num_positives > 1 else []
                
                # Compute base CLIP similarities
                with torch.no_grad():
                    # Process image
                    if self.is_ft:
                        # FT mode: Use already preprocessed tensor
                        img_input = image_tensor.unsqueeze(0).to(self.device)
                        image_features = self.base_clip_model.encode_image(img_input)
                    else:
                        # Standard mode: Preprocess PIL image
                        img_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
                        output = self.base_clip_model(img_input, None)
                        if isinstance(output, tuple):
                            image_features = output[0]
                        else:
                            image_features = output['image_features']
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Process full caption
                    text_tokens = torch.cat([
                        clip.tokenize(full_caption, truncate=True)
                    ]).to(self.device)
                    
                    if self.is_ft:
                        text_features = self.base_clip_model.encode_text(text_tokens)
                    else:
                        output = self.base_clip_model(None, text_tokens)
                        if isinstance(output, tuple):
                            text_features = output[1]
                        else:
                            text_features = output['text_features']
                    
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarity for full caption
                    base_sim_full = (image_features @ text_features.T).squeeze().item()
                    
                    # Compute similarities for components
                    base_sim_components = []
                    for comp_caption in component_captions:
                        comp_tokens = torch.cat([
                            clip.tokenize(comp_caption, truncate=True)
                        ]).to(self.device)
                        
                        if self.is_ft:
                            comp_features = self.base_clip_model.encode_text(comp_tokens)
                        else:
                            output = self.base_clip_model(None, comp_tokens)
                            if isinstance(output, tuple):
                                comp_features = output[1]
                            else:
                                comp_features = output['text_features']
                        
                        comp_features = comp_features / comp_features.norm(dim=-1, keepdim=True)
                        comp_sim = (image_features @ comp_features.T).squeeze().item()
                        base_sim_components.append(comp_sim)
                    
                    # Compute similarities for negative captions
                    base_sim_negatives = []
                    for neg_caption in negative_captions:
                        neg_tokens = torch.cat([
                            clip.tokenize(neg_caption, truncate=True)
                        ]).to(self.device)
                        
                        if self.is_ft:
                            neg_features = self.base_clip_model.encode_text(neg_tokens)
                        else:
                            output = self.base_clip_model(None, neg_tokens)
                            if isinstance(output, tuple):
                                neg_features = output[1]
                            else:
                                neg_features = output['text_features']
                        
                        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
                        neg_sim = (image_features @ neg_features.T).squeeze().item()
                        base_sim_negatives.append(neg_sim)
                
                # Store sample data
                self.samples.append({
                    'index': idx,
                    'image': image_pil,
                    'full_caption': full_caption,
                    'component_captions': component_captions,
                    'negative_captions': negative_captions,
                    'base_sim_full': base_sim_full,
                    'base_sim_components': base_sim_components,
                    'base_sim_negatives': base_sim_negatives,
                    'image_tensor': image_tensor  # Store for model inference
                })
                
            except Exception as e:
                logging.warning(f"Failed to extract sample {idx}: {e}")
                import traceback
                logging.warning(f"Traceback: {traceback.format_exc()}")
                continue
    
    def _wrap_text(self, text: str, width: int = 30) -> str:
        """Wrap text to specified width for better display."""
        lines = textwrap.wrap(text, width=width)
        return '\n'.join(lines[:5])  # Limit to 5 lines
    
    def log_epoch_samples(
        self, 
        model, 
        epoch: int, 
        wandb_prefix: str = "train_samples",
        log_images: bool = True,
        log_metrics: bool = True
    ) -> Dict[str, any]:
        """
        Compute sample similarities for current epoch and prepare logging data.
        
        Returns metrics and images to be logged by caller at the same step as training metrics.
        
        Note: Images are only logged on the first call (first epoch). Subsequent calls
        only log similarity metrics as tables/scalars to save bandwidth and storage.
        
        Args:
            model: Current model being trained
            epoch: Current epoch number
            wandb_prefix: Prefix for wandb logging keys
            log_images: Whether to prepare visualization images (only effective on first call)
            log_metrics: Whether to compute similarity metrics
            
        Returns:
            Dict with:
                - 'metrics': Dict of metrics to log (if log_metrics=True)
                - 'images': List of wandb.Image objects (only on first call if log_images=True)
                - 'tables': List of wandb.Table objects with similarity data
                - 'wandb_prefix': The prefix to use for logging
        """
        if len(self.samples) == 0:
            logging.warning("No samples to log")
            return {}
        
        # Compute current model similarities
        current_sims = self._compute_current_similarities(model)
        
        result = {
            'wandb_prefix': wandb_prefix,
            'epoch': epoch
        }
        
        # Create visualization with images only on first call
        if log_images and not self.images_logged:
            viz_images = self._create_visualization(current_sims, epoch)
            
            # Prepare wandb images
            wandb_images = []
            for i, (img, caption) in enumerate(viz_images):
                wandb_images.append(
                    wandb.Image(
                        img, 
                        caption=f"Sample {i+1}: {caption[:100]}..."
                    )
                )
            
            result['images'] = wandb_images
            self.images_logged = True
            logging.info(f"Logged {len(self.samples)} sample images for epoch {epoch} (first time only)")
        
        # Always create similarity tables (lightweight)
        if log_metrics:
            metrics = self._compute_metrics(current_sims)
            result['metrics'] = metrics
            
            # Create similarity tables for wandb
            tables = self._create_similarity_tables(current_sims, epoch)
            result['tables'] = tables
        
        logging.info(f"Prepared {len(self.samples)} samples for epoch {epoch}")
        
        return result
    
    def _compute_current_similarities(self, model) -> List[Dict]:
        """Compute similarities using current model."""
        results = []
        
        with torch.no_grad():
            for sample in self.samples:
                # Get image features
                image_tensor = sample['image_tensor'].unsqueeze(0).to(self.device)
                
                if self.is_ft:
                    image_features = model.encode_image(image_tensor)
                else:
                    output = model(image_tensor, None)
                    if isinstance(output, tuple):
                        image_features = output[0]
                    else:
                        image_features = output['image_features']
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Full caption similarity
                text_tokens = torch.cat([
                    clip.tokenize(sample['full_caption'], truncate=True)
                ]).to(self.device)
                
                if self.is_ft:
                    text_features = model.encode_text(text_tokens)
                else:
                    output = model(None, text_tokens)
                    if isinstance(output, tuple):
                        text_features = output[1]
                    else:
                        text_features = output['text_features']
                
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                current_sim_full = (image_features @ text_features.T).squeeze().item()
                
                # Component similarities
                current_sim_components = []
                for comp_caption in sample['component_captions']:
                    comp_tokens = torch.cat([
                        clip.tokenize(comp_caption, truncate=True)
                    ]).to(self.device)
                    
                    if self.is_ft:
                        comp_features = model.encode_text(comp_tokens)
                    else:
                        output = model(None, comp_tokens)
                        if isinstance(output, tuple):
                            comp_features = output[1]
                        else:
                            comp_features = output['text_features']
                    
                    comp_features = comp_features / comp_features.norm(dim=-1, keepdim=True)
                    comp_sim = (image_features @ comp_features.T).squeeze().item()
                    current_sim_components.append(comp_sim)
                
                # Negative similarities
                current_sim_negatives = []
                for neg_caption in sample['negative_captions']:
                    neg_tokens = torch.cat([
                        clip.tokenize(neg_caption, truncate=True)
                    ]).to(self.device)
                    
                    if self.is_ft:
                        neg_features = model.encode_text(neg_tokens)
                    else:
                        output = model(None, neg_tokens)
                        if isinstance(output, tuple):
                            neg_features = output[1]
                        else:
                            neg_features = output['text_features']
                    
                    neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
                    neg_sim = (image_features @ neg_features.T).squeeze().item()
                    current_sim_negatives.append(neg_sim)
                
                results.append({
                    'current_sim_full': current_sim_full,
                    'current_sim_components': current_sim_components,
                    'current_sim_negatives': current_sim_negatives
                })
        
        return results
    
    def _create_visualization(self, current_sims: List[Dict], epoch: int) -> List[Tuple[Image.Image, str]]:
        """Create visualization using a grid layout that prevents caption collisions."""
        viz_images = []
        
        for i, (sample, curr_sim) in enumerate(zip(self.samples, current_sims)):
            # Limit number of components to display
            num_components = min(len(sample['component_captions']), self.max_components)
            num_negatives = len(sample['negative_captions'])
            total_components = len(sample['component_captions'])
            
            # Create items: [image, full, comp1, comp2, ..., neg1, neg2, neg3]
            items = []
            
            # Image item
            items.append({
                'type': 'image',
                'title': 'Image',
                'image': sample['image']
            })
            
            # Full caption item
            items.append({
                'type': 'full',
                'title': 'Full Caption',
                'base': sample['base_sim_full'],
                'curr': curr_sim['current_sim_full'],
                'delta': curr_sim['current_sim_full'] - sample['base_sim_full'],
                'caption': sample['full_caption']
            })
            
            # Component items
            for j in range(num_components):
                title = f'Component {j+1}'
                if j == num_components - 1 and total_components > num_components:
                    title += f' (+{total_components - num_components})'
                items.append({
                    'type': 'component',
                    'title': title,
                    'base': sample['base_sim_components'][j],
                    'curr': curr_sim['current_sim_components'][j],
                    'delta': curr_sim['current_sim_components'][j] - sample['base_sim_components'][j],
                    'caption': sample['component_captions'][j]
                })
            
            # Negative items
            for j in range(num_negatives):
                items.append({
                    'type': 'negative',
                    'title': f'Negative {j+1}',
                    'base': sample['base_sim_negatives'][j],
                    'curr': curr_sim['current_sim_negatives'][j],
                    'delta': curr_sim['current_sim_negatives'][j] - sample['base_sim_negatives'][j],
                    'caption': sample['negative_captions'][j]
                })
            
            # Calculate grid dimensions
            total_items = len(items)
            grid_cols = min(4, total_items)  # Max 4 columns
            grid_rows = (total_items + grid_cols - 1) // grid_cols  # Ceiling division
            
            # Create figure with subplots
            fig_width = grid_cols * 4
            fig_height = grid_rows * 3.5
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
            
            # Handle single row/col case
            if grid_rows == 1 and grid_cols == 1:
                axes = np.array([[axes]])
            elif grid_rows == 1:
                axes = axes.reshape(1, -1)
            elif grid_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Fill grid with items
            for idx, item in enumerate(items):
                row = idx // grid_cols
                col = idx % grid_cols
                ax = axes[row, col]
                
                if item['type'] == 'image':
                    # Draw image
                    ax.imshow(item['image'])
                    ax.axis('off')
                    ax.set_title(item['title'], fontsize=12, fontweight='bold', pad=10)
                else:
                    # Draw caption card
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    
                    # Determine color based on delta
                    if item['type'] == 'negative':
                        # For negatives, negative delta is good (green)
                        color = 'green' if item['delta'] < 0 else 'red'
                        title_color = 'darkred'
                    else:
                        # For positives, positive delta is good (green)
                        color = 'green' if item['delta'] > 0 else 'red'
                        title_color = 'black'
                    
                    # Title at top
                    ax.text(0.5, 0.95, item['title'], 
                           ha='center', va='top', fontsize=11, fontweight='bold',
                           color=title_color)
                    
                    # Metrics box in middle - create a background rectangle first
                    metrics_y_center = 0.65
                    metrics_height = 0.25
                    metrics_rect = plt.Rectangle((0.1, metrics_y_center - metrics_height/2), 0.8, metrics_height,
                                                 facecolor='lightgray', edgecolor=color, 
                                                 linewidth=2, zorder=1)
                    ax.add_patch(metrics_rect)
                    
                    # Metrics text on top of box
                    ax.text(0.5, 0.72, f"Base: {item['base']:.3f}",
                           ha='center', va='center', fontsize=9, zorder=2)
                    ax.text(0.5, 0.65, f"Curr: {item['curr']:.3f}",
                           ha='center', va='center', fontsize=9, zorder=2)
                    ax.text(0.5, 0.58, f"Δ: {item['delta']:+.3f}",
                           ha='center', va='center', fontsize=9, fontweight='bold', zorder=2)
                    
                    # Caption text at bottom - word wrapped, line by line
                    caption = item['caption']
                    wrapped_lines = textwrap.wrap(caption, width=35)[:6]  # Max 6 lines
                    
                    # Calculate starting y position for caption text
                    line_spacing = 0.06
                    start_y = 0.35
                    
                    for line_idx, line in enumerate(wrapped_lines):
                        y_pos = start_y - (line_idx * line_spacing)
                        ax.text(0.5, y_pos, line,
                               ha='center', va='top', fontsize=8, style='italic')
                    
                    # Add border around entire cell
                    border_rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                               fill=False, edgecolor='gray', 
                                               linewidth=1.5)
                    ax.add_patch(border_rect)
            
            # Hide empty subplots
            for idx in range(total_items, grid_rows * grid_cols):
                row = idx // grid_cols
                col = idx % grid_cols
                axes[row, col].axis('off')
            
            # Add sample info as suptitle
            plt.suptitle(f"Sample {i+1} | Epoch {epoch}", fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            plt.tight_layout()
            
            # Convert to PIL
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            
            viz_images.append((img, sample['full_caption']))
            plt.close(fig)
        
        return viz_images
    
    def _create_similarity_tables(self, current_sims: List[Dict], epoch: int) -> List[wandb.Table]:
        """
        Create wandb tables with similarity scores (lightweight, no images).
        
        This creates compact tables showing:
        - Sample ID, Caption Type, Base Sim, Current Sim, Delta
        
        Returns:
            List of wandb.Table objects for logging
        """
        tables = []
        
        # Create a single table with all samples and their caption similarities
        columns = ["Sample", "Type", "Caption", "Base Sim", "Current Sim", "Delta (Δ)"]
        data = []
        
        for i, (sample, curr_sim) in enumerate(zip(self.samples, current_sims)):
            sample_id = f"Sample {i+1}"
            
            # Full caption row
            data.append([
                sample_id,
                "Full",
                self._truncate_text(sample['full_caption'], 60),
                f"{sample['base_sim_full']:.3f}",
                f"{curr_sim['current_sim_full']:.3f}",
                f"{curr_sim['current_sim_full'] - sample['base_sim_full']:+.3f}"
            ])
            
            # Component rows
            for j, (comp_caption, base_comp, curr_comp) in enumerate(zip(
                sample['component_captions'],
                sample['base_sim_components'],
                curr_sim['current_sim_components']
            )):
                data.append([
                    sample_id,
                    f"Comp {j+1}",
                    self._truncate_text(comp_caption, 60),
                    f"{base_comp:.3f}",
                    f"{curr_comp:.3f}",
                    f"{curr_comp - base_comp:+.3f}"
                ])
            
            # Negative rows
            for j, (neg_caption, base_neg, curr_neg) in enumerate(zip(
                sample['negative_captions'],
                sample['base_sim_negatives'],
                curr_sim['current_sim_negatives']
            )):
                data.append([
                    sample_id,
                    f"Neg {j+1}",
                    self._truncate_text(neg_caption, 60),
                    f"{base_neg:.3f}",
                    f"{curr_neg:.3f}",
                    f"{curr_neg - base_neg:+.3f}"
                ])
        
        table = wandb.Table(columns=columns, data=data)
        tables.append(table)
        
        return tables
    
    def _truncate_text(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."
    
    def _compute_metrics(self, current_sims: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics across all samples."""
        metrics = {}
        
        # Average improvements
        full_improvements = []
        comp_improvements = []
        neg_improvements = []
        
        for i, (sample, curr_sim) in enumerate(zip(self.samples, current_sims)):
            # Full caption
            base_full = sample['base_sim_full']
            curr_full = curr_sim['current_sim_full']
            full_improvements.append(curr_full - base_full)
            
            # Components
            for base_comp, curr_comp in zip(
                sample['base_sim_components'],
                curr_sim['current_sim_components']
            ):
                comp_improvements.append(curr_comp - base_comp)
            
            # Negatives (negative delta is good - decreasing similarity)
            for base_neg, curr_neg in zip(
                sample['base_sim_negatives'],
                curr_sim['current_sim_negatives']
            ):
                neg_improvements.append(base_neg - curr_neg)  # Note: inverted for negatives
            
            # Per-sample metrics
            metrics[f'sample_{i+1}/full_sim'] = curr_full
            metrics[f'sample_{i+1}/full_improvement'] = curr_full - base_full
            
            # Average component similarity for this sample
            if len(curr_sim['current_sim_components']) > 0:
                avg_comp_sim = sum(curr_sim['current_sim_components']) / len(curr_sim['current_sim_components'])
                metrics[f'sample_{i+1}/avg_component_sim'] = avg_comp_sim
            
            # Average negative similarity for this sample
            if len(curr_sim['current_sim_negatives']) > 0:
                avg_neg_sim = sum(curr_sim['current_sim_negatives']) / len(curr_sim['current_sim_negatives'])
                metrics[f'sample_{i+1}/avg_negative_sim'] = avg_neg_sim
                metrics[f'sample_{i+1}/avg_negative_improvement'] = sum([
                    sample['base_sim_negatives'][j] - curr_sim['current_sim_negatives'][j]
                    for j in range(len(curr_sim['current_sim_negatives']))
                ]) / len(curr_sim['current_sim_negatives'])
        
        # Aggregate metrics
        if full_improvements:
            metrics['avg_full_improvement'] = np.mean(full_improvements)
            metrics['avg_full_sim'] = np.mean([s['current_sim_full'] for s in current_sims])
        
        if comp_improvements:
            metrics['avg_comp_improvement'] = np.mean(comp_improvements)
            all_comp_sims = [
                sim for curr_sim in current_sims 
                for sim in curr_sim['current_sim_components']
            ]
            metrics['avg_comp_sim'] = np.mean(all_comp_sims)
        
        if neg_improvements:
            # For negatives, positive improvement means similarity decreased (good!)
            metrics['avg_neg_improvement'] = np.mean(neg_improvements)
            all_neg_sims = [
                sim for curr_sim in current_sims 
                for sim in curr_sim['current_sim_negatives']
            ]
            metrics['avg_neg_sim'] = np.mean(all_neg_sims)
        
        return metrics
