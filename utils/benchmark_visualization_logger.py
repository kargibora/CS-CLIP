"""
Benchmark Visualization Logger for Dataset Evaluation

This module provides functionality to visualize random samples from benchmark datasets
during evaluation. It shows how model similarities evolve across training epochs.

Key Features:
  - Samples random images/captions from each benchmark dataset
  - Visualizes all caption options (positive and negative) with similarity scores
  - Tracks similarity evolution across epochs
  - Memory-efficient: logs images only once per dataset, then only metrics
  - Supports all dataset types through universal __getitem__ interface
  - Wandb integration with images and metrics
"""

import logging
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import clip
import wandb
from PIL import Image
from typing import Dict, List, Tuple, Optional
import traceback


class BenchmarkVisualizationLogger:
    """
    Logger for visualizing random samples from benchmark datasets during evaluation.
    
    This class:
      1. Samples random items from each benchmark dataset
      2. Extracts images and all caption options (positive/negative)
      3. Computes similarities for each caption option
      4. Creates visualization panels showing images and similarities
      5. Logs to wandb with images (first call only) and metrics (always)
      
    Memory Optimization:
      - Images are logged only on the first call per dataset
      - Subsequent calls only log metrics to save memory
      - Use reset_image_logging() to re-enable image logging if needed
    """
    
    def __init__(
        self,
        device: torch.device,
        num_samples_per_dataset: int = 5,
        seed: int = 42,
        is_ft: bool = False
    ):
        """
        Initialize the benchmark visualization logger.
        
        Args:
            device: Device for computation
            num_samples_per_dataset: Number of samples to visualize per dataset (default: 5)
            seed: Random seed for sample selection
            is_ft: Whether fine-tuning mode is active
        """
        self.device = device
        self.num_samples_per_dataset = num_samples_per_dataset
        self.seed = seed
        self.is_ft = is_ft
        
        # Store samples per dataset
        self.dataset_samples = {}  # {dataset_name: [sample_data]}
        
        # Track which datasets have already had images logged (memory efficiency)
        self.images_logged_for_dataset = set()  # {dataset_key}
        
        logging.info(f"BenchmarkVisualizationLogger: Initialized with {num_samples_per_dataset} samples per dataset")
        logging.info("Images will only be logged once per dataset to save memory")
    
    def sample_dataset(
        self, 
        dataset, 
        dataset_name: str, 
        subset_name: str,
        preprocess
    ):
        """
        Sample random items from a dataset for visualization.
        
        Args:
            dataset: Dataset object with __getitem__ method
            dataset_name: Name of the dataset
            subset_name: Name of the subset
            preprocess: CLIP preprocessing function
        """
        key = f"{dataset_name}/{subset_name}"
        
        # Skip if already sampled
        if key in self.dataset_samples:
            return
        
        # Set seed based on dataset/subset name for different samples per subset
        # Use hash of the key to get a deterministic but unique seed per subset
        subset_seed = self.seed + hash(key) % 10000
        np.random.seed(subset_seed)
        
        dataset_size = len(dataset)
        num_samples = min(self.num_samples_per_dataset, dataset_size)
        
        if dataset_size == 0:
            logging.warning(f"Dataset {key} is empty, skipping sampling")
            return
        
        sample_indices = np.random.choice(dataset_size, size=num_samples, replace=False).tolist()
        
        samples = []
        for idx in sample_indices:
            try:
                item = dataset[idx]
                
                # Extract image and captions based on item structure
                sample_data = self._extract_sample_data(item, idx, preprocess)
                
                if sample_data is not None:
                    samples.append(sample_data)
                    
            except Exception as e:
                logging.warning(f"Failed to extract sample {idx} from {key}: {e}")
                continue
        
        if samples:
            self.dataset_samples[key] = samples
            logging.info(f"Sampled {len(samples)} items from {key}")
    
    def _extract_sample_data(self, item: Dict, idx: int, preprocess) -> Optional[Dict]:
        """
        Extract image and caption data from a dataset item.
        
        Supports various dataset formats:
        - image_options + caption_options (multi-caption format)
        - image + caption (single caption format)
        - images + captions (batch format)
        """
        try:
            # Handle multi-caption format (most common in benchmarks)
            if 'image_options' in item and 'caption_options' in item:
                image_tensor = item['image_options']
                captions = item['caption_options']
                num_positives = item.get('num_positives', 1)
                
                # Split into positive and negative captions
                positive_captions = captions[:num_positives]
                negative_captions = captions[num_positives:]
                
                # Denormalize image for visualization
                image_pil = self._denormalize_image(image_tensor)
                
                return {
                    'index': idx,
                    'image': image_pil,
                    'image_tensor': image_tensor,
                    'positive_captions': positive_captions,
                    'negative_captions': negative_captions,
                    'all_captions': captions
                }
            
            # Handle single caption format
            elif 'image' in item and 'caption' in item:
                image_tensor = item['image']
                caption = item['caption']
                
                # Check for negative captions
                negative_caption = item.get('negative_caption', None)
                
                image_pil = self._denormalize_image(image_tensor)
                
                positive_captions = [caption]
                negative_captions = [negative_caption] if negative_caption else []
                all_captions = positive_captions + negative_captions
                
                return {
                    'index': idx,
                    'image': image_pil,
                    'image_tensor': image_tensor,
                    'positive_captions': positive_captions,
                    'negative_captions': negative_captions,
                    'all_captions': all_captions
                }
            
            # Handle batch format (images + captions)
            elif 'images' in item and 'captions' in item:
                # Take first image from batch
                image_tensor = item['images'][0] if isinstance(item['images'], torch.Tensor) else item['images']
                captions = item['captions']
                
                if isinstance(captions, list):
                    all_captions = captions
                else:
                    all_captions = [captions]
                
                image_pil = self._denormalize_image(image_tensor)
                
                return {
                    'index': idx,
                    'image': image_pil,
                    'image_tensor': image_tensor,
                    'positive_captions': all_captions[:1],
                    'negative_captions': all_captions[1:],
                    'all_captions': all_captions
                }
            
            else:
                logging.warning(f"Unknown item format with keys: {item.keys()}")
                return None
                
        except Exception as e:
            logging.warning(f"Error extracting sample data: {e}")
            logging.warning(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _denormalize_image(self, image_tensor: torch.Tensor) -> Image.Image:
        """Denormalize image tensor and convert to PIL Image."""
        # Move mean and std to the same device as image_tensor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device).view(3, 1, 1)
        
        image_denorm = image_tensor * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        image_pil = Image.fromarray(
            (image_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        
        return image_pil
    
    def compute_similarities(
        self,
        model,
        clip_model,
        dataset_name: str,
        subset_name: str,
        epoch: int
    ) -> Dict[str, any]:
        """
        Compute similarities for sampled items and create visualizations.
        
        Args:
            model: Current model being evaluated
            clip_model: Base CLIP model for comparison
            dataset_name: Name of the dataset
            subset_name: Name of the subset
            epoch: Current epoch number
            
        Returns:
            Dict with wandb images and metrics
        """
        key = f"{dataset_name}/{subset_name}"
        
        if key not in self.dataset_samples:
            logging.warning(f"No samples available for {key}")
            return {}
        
        samples = self.dataset_samples[key]
        
        # Compute similarities for all samples
        results = []
        for sample in samples:
            sample_result = self._compute_sample_similarities(
                sample, model, clip_model
            )
            if sample_result:
                results.append(sample_result)
        
        if not results:
            logging.warning(f"No valid results for {key}")
            return {}
        
        # Check if we should log images for this dataset (only on first call)
        should_log_images = key not in self.images_logged_for_dataset
        
        # Create visualizations only if we need to log images
        wandb_images = []
        if should_log_images:
            viz_images = self._create_visualizations(results, samples, epoch, key)
            
            # Prepare wandb images
            for img, caption in viz_images:
                wandb_images.append(
                    wandb.Image(img, caption=f"{key}: {caption[:100]}...")
                )
            
            # Mark this dataset as having images logged
            self.images_logged_for_dataset.add(key)
            logging.info(f"Images logged for {key} (first time). Future calls will only log metrics.")
        else:
            logging.debug(f"Skipping image logging for {key} (already logged). Only computing metrics.")
        
        # Compute aggregate metrics (always do this)
        metrics = self._compute_aggregate_metrics(results, key)
        
        return {
            'images': wandb_images,  # Will be empty list if not first call
            'metrics': metrics,
            'key': key
        }
    
    def _compute_sample_similarities(
        self,
        sample: Dict,
        model,
        clip_model
    ) -> Optional[Dict]:
        """Compute similarities for all captions in a sample."""
        try:
            with torch.no_grad():
                # Get image features
                image_tensor = sample['image_tensor'].unsqueeze(0).to(self.device)
                
                # Compute image features for both models
                if self.is_ft:
                    current_img_features = model.encode_image(image_tensor)
                    base_img_features = clip_model.encode_image(image_tensor)
                else:
                    # For LabCLIP models
                    current_output = model(image_tensor, None)
                    if isinstance(current_output, tuple):
                        current_img_features = current_output[0]
                    elif isinstance(current_output, dict):
                        # Handle TQA dict output
                        current_img_features = current_output.get('image_embeds', current_output.get('image'))
                    else:
                        current_img_features = current_output
                    
                    base_img_features = clip_model.encode_image(image_tensor)
                
                # Handle dict outputs (e.g., from TQA models)
                if isinstance(current_img_features, dict):
                    current_img_features = current_img_features.get('image_embeds', current_img_features.get('global'))
                
                current_img_features = current_img_features / current_img_features.norm(dim=-1, keepdim=True)
                base_img_features = base_img_features / base_img_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities for all captions
                all_captions = sample['all_captions']
                num_positives = len(sample['positive_captions'])
                
                current_sims = []
                base_sims = []
                
                for caption in all_captions:
                    text_tokens = clip.tokenize([caption], truncate=True).to(self.device)
                    
                    # Current model text features
                    if self.is_ft:
                        current_text_features = model.encode_text(text_tokens)
                    else:
                        current_output = model(None, text_tokens)
                        if isinstance(current_output, tuple):
                            current_text_features = current_output[1]
                        elif isinstance(current_output, dict):
                            # Handle TQA dict output
                            current_text_features = current_output.get('text_embeds', current_output.get('text'))
                        else:
                            current_text_features = current_output
                    
                    # Handle dict outputs (e.g., from TQA models)
                    if isinstance(current_text_features, dict):
                        current_text_features = current_text_features.get('text_embeds', current_text_features.get('global'))
                    
                    # Base model text features
                    base_text_features = clip_model.encode_text(text_tokens)
                    
                    current_text_features = current_text_features / current_text_features.norm(dim=-1, keepdim=True)
                    base_text_features = base_text_features / base_text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    current_sim = (current_img_features @ current_text_features.T).squeeze().item()
                    base_sim = (base_img_features @ base_text_features.T).squeeze().item()
                    
                    current_sims.append(current_sim)
                    base_sims.append(base_sim)
                
                return {
                    'current_sims': current_sims,
                    'base_sims': base_sims,
                    'num_positives': num_positives
                }
                
        except Exception as e:
            logging.warning(f"Error computing similarities: {e}")
            logging.warning(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _create_visualizations(
        self,
        results: List[Dict],
        samples: List[Dict],
        epoch: int,
        dataset_key: str
    ) -> List[Tuple[Image.Image, str]]:
        """Create visualization panels for samples."""
        viz_images = []
        
        for i, (result, sample) in enumerate(zip(results, samples)):
            try:
                fig = self._create_sample_panel(result, sample, epoch, dataset_key, i + 1)
                
                # Convert to PIL Image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf).copy()
                buf.close()
                
                viz_images.append((img, sample['positive_captions'][0]))
                plt.close(fig)
                
            except Exception as e:
                logging.warning(f"Error creating visualization for sample {i}: {e}")
                continue
        
        return viz_images
    
    def _create_sample_panel(
        self,
        result: Dict,
        sample: Dict,
        epoch: int,
        dataset_key: str,
        sample_num: int
    ) -> plt.Figure:
        """Create a visualization panel for a single sample."""
        all_captions = sample['all_captions']
        num_positives = result['num_positives']
        current_sims = result['current_sims']
        base_sims = result['base_sims']
        
        # Calculate grid: image on left, captions on right
        num_captions = len(all_captions)
        
        # Create figure with custom layout
        fig_width = 14
        fig_height = max(6, num_captions * 1.5)
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create grid spec: 1 column for image, rest for captions
        gs = fig.add_gridspec(max(num_captions, 4), 2, width_ratios=[1, 2], hspace=0.3, wspace=0.3)
        
        # Image subplot (spans multiple rows)
        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(sample['image'])
        ax_img.axis('off')
        ax_img.set_title('Image', fontsize=12, fontweight='bold')
        
        # Caption subplots
        for idx, (caption, curr_sim, base_sim) in enumerate(zip(all_captions, current_sims, base_sims)):
            ax = fig.add_subplot(gs[idx, 1])
            ax.axis('off')
            
            # Determine caption type
            is_positive = idx < num_positives
            caption_type = "Positive" if is_positive else "Negative"
            color = 'green' if is_positive else 'red'
            
            # Compute delta
            delta = curr_sim - base_sim
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            
            # Truncate caption
            caption_display = caption[:100] + "..." if len(caption) > 100 else caption
            
            # Create text display
            text_content = f"{caption_type} Caption {idx + 1}:\n"
            text_content += f"{caption_display}\n\n"
            text_content += f"Base CLIP: {base_sim:.3f}\n"
            text_content += f"Current: {curr_sim:.3f}\n"
            text_content += f"Δ: {delta_str}"
            
            # Add text with colored border
            ax.text(
                0.05, 0.95, text_content,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor=color, linewidth=2)
            )
        
        # Add title
        plt.suptitle(
            f"{dataset_key} - Sample {sample_num} | Epoch {epoch}",
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        
        return fig
    
    def _compute_aggregate_metrics(
        self,
        results: List[Dict],
        dataset_key: str
    ) -> Dict[str, float]:
        """Compute aggregate metrics across samples."""
        metrics = {}
        
        all_pos_current = []
        all_pos_base = []
        all_pos_deltas = []
        
        all_neg_current = []
        all_neg_base = []
        all_neg_deltas = []
        
        for result in results:
            num_pos = result['num_positives']
            current_sims = result['current_sims']
            base_sims = result['base_sims']
            
            # Positive captions
            pos_current = current_sims[:num_pos]
            pos_base = base_sims[:num_pos]
            all_pos_current.extend(pos_current)
            all_pos_base.extend(pos_base)
            all_pos_deltas.extend([c - b for c, b in zip(pos_current, pos_base)])
            
            # Negative captions
            if len(current_sims) > num_pos:
                neg_current = current_sims[num_pos:]
                neg_base = base_sims[num_pos:]
                all_neg_current.extend(neg_current)
                all_neg_base.extend(neg_base)
                # For negatives, we want similarity to decrease, so invert delta
                all_neg_deltas.extend([b - c for c, b in zip(neg_current, neg_base)])
        
        # Compute metrics
        if all_pos_current:
            metrics[f'{dataset_key}/viz_avg_positive_sim'] = np.mean(all_pos_current)
            metrics[f'{dataset_key}/viz_avg_positive_improvement'] = np.mean(all_pos_deltas)
        
        if all_neg_current:
            metrics[f'{dataset_key}/viz_avg_negative_sim'] = np.mean(all_neg_current)
            metrics[f'{dataset_key}/viz_avg_negative_improvement'] = np.mean(all_neg_deltas)
        
        # Overall improvement
        all_deltas = all_pos_deltas + all_neg_deltas
        if all_deltas:
            metrics[f'{dataset_key}/viz_avg_overall_improvement'] = np.mean(all_deltas)
        
        return metrics
    
    def log_all_datasets(
        self,
        model,
        clip_model,
        epoch: int,
        wandb_prefix: str = "benchmark_viz"
    ) -> Dict[str, any]:
        """
        Compute similarities and create visualizations for all sampled datasets.
        
        Args:
            model: Current model being evaluated
            clip_model: Base CLIP model
            epoch: Current epoch number
            wandb_prefix: Prefix for wandb logging
            
        Returns:
            Dict with all images and metrics to log
        """
        all_images = []
        all_metrics = {}
        
        for dataset_key in self.dataset_samples.keys():
            dataset_name, subset_name = dataset_key.split('/', 1)
            
            result = self.compute_similarities(
                model=model,
                clip_model=clip_model,
                dataset_name=dataset_name,
                subset_name=subset_name,
                epoch=epoch
            )
            
            if result:
                all_images.extend(result.get('images', []))
                all_metrics.update(result.get('metrics', {}))
        
        return {
            'images': all_images,
            'metrics': all_metrics,
            'wandb_prefix': wandb_prefix,
            'epoch': epoch
        }
    
    def reset_image_logging(self, dataset_key: Optional[str] = None):
        """
        Reset image logging state to allow images to be logged again.
        
        Args:
            dataset_key: Specific dataset key to reset (e.g., "VALSE/existence").
                        If None, resets all datasets.
        """
        if dataset_key is None:
            # Reset all datasets
            num_reset = len(self.images_logged_for_dataset)
            self.images_logged_for_dataset.clear()
            logging.info(f"Reset image logging for all {num_reset} datasets")
        else:
            # Reset specific dataset
            if dataset_key in self.images_logged_for_dataset:
                self.images_logged_for_dataset.remove(dataset_key)
                logging.info(f"Reset image logging for {dataset_key}")
            else:
                logging.warning(f"Dataset {dataset_key} was not in logged state")
    
    def get_logged_datasets(self) -> List[str]:
        """
        Get list of dataset keys that have already had images logged.
        
        Returns:
            List of dataset keys (e.g., ["VALSE/existence", "BLA/ap"])
        """
        return list(self.images_logged_for_dataset)
