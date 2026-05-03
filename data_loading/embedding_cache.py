"""
Universal embedding caching system for all evaluation datasets.

This module provides functionality to cache and load image/text embeddings 
to skip CLIP computations for faster evaluation.
"""

import hashlib
import logging
import os
from typing import Any, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def get_cache_path(dataset_name: str, subset_name: str, embedding_type: str, cache_dir: str = "./cache") -> str:
    """
    Generate standardized cache path for embeddings.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'VALSE', 'BLA', etc.)
        subset_name: Subset name (e.g., 'existence', 'action', etc.)
        embedding_type: 'text' or 'image'
        cache_dir: Base cache directory
        
    Returns:
        Path to cache file
    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{dataset_name}_{subset_name}_{embedding_type}_embeddings.pt"
    return os.path.join(cache_dir, filename)


def should_use_cache(embedding_model, aligning_model=None) -> bool:
    """
    Determine if caching should be used based on the model type.
    
    Args:
        embedding_model: The CLIP embedding model
        aligning_model: Optional alignment model
        
    Returns:
        True if caching should be used, False otherwise
        
    Notes:
        - For CLIP + no alignment: Cache final embeddings
        - For CLIP + simple alignment: Cache CLIP embeddings, apply alignment on-the-fly
        - For CLIPMultiLayerFTAlignment: Don't use cache (embeddings depend on fine-tuning)
    """

    # Check if this is CLIPMultiLayerFTAlignment - don't use cache in this case
    if aligning_model is not None:
        return True
    
    if embedding_model is not None:
        return False
    

def get_model_hash(embedding_model, aligning_model=None) -> str:
    """
    Generate a hash for the model configuration to ensure cache compatibility.
    
    Args:
        embedding_model: The CLIP embedding model
        aligning_model: Optional alignment model
        
    Returns:
        Hash string representing the model configuration
    """
    # Create a simple hash based on model class and key parameters
    model_info = f"{embedding_model.__class__.__name__}"
    
    if aligning_model is not None:
        model_info += f"_{aligning_model.__class__.__name__}"
    
    return hashlib.md5(model_info.encode()).hexdigest()[:8]


def load_cached_embeddings(
    dataset_name: str, 
    subset_name: str, 
    embedding_type: str,
    device: str = "cuda",
    cache_dir: str = "./cache"
) -> Optional[torch.Tensor]:
    """
    Load cached embeddings if they exist.
    
    Args:
        dataset_name: Name of the dataset
        subset_name: Subset name  
        embedding_type: 'text' or 'image'
        device: Device to load embeddings to
        cache_dir: Base cache directory
        
    Returns:
        Cached embeddings tensor or None if not found
    """
    cache_path = get_cache_path(dataset_name, subset_name, embedding_type, cache_dir)
    
    if os.path.exists(cache_path):
        try:
            embeddings = torch.load(cache_path, map_location=device)
            logger.info(f"Loaded cached {embedding_type} embeddings from {cache_path}")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings from {cache_path}: {e}")
            return None
    
    return None


def save_cached_embeddings(
    embeddings: torch.Tensor,
    dataset_name: str,
    subset_name: str, 
    embedding_type: str,
    cache_dir: str = "./cache"
) -> None:
    """
    Save embeddings to cache.
    
    Args:
        embeddings: Embeddings tensor to save
        dataset_name: Name of the dataset
        subset_name: Subset name
        embedding_type: 'text' or 'image' 
        cache_dir: Base cache directory
    """
    cache_path = get_cache_path(dataset_name, subset_name, embedding_type, cache_dir)
    
    try:
        torch.save(embeddings, cache_path)
        logger.info(f"Saved {embedding_type} embeddings to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save embeddings to {cache_path}: {e}")


def compute_or_load_embeddings(
    data_batch: List[Any],
    embedding_model,
    aligning_model,
    dataset_name: str,
    subset_name: str,
    embedding_type: str,
    device: str = "cuda",
    cache_dir: str = "./cache",
    intermediate_layer_names: List[str] = ["final"],
    batch_idx: int = 0,
    compute_func: Optional[callable] = None
) -> torch.Tensor:
    """
    Universal function to compute or load embeddings with caching.
    
    Args:
        data_batch: Batch of data to compute embeddings for
        embedding_model: CLIP embedding model
        aligning_model: Optional alignment model
        dataset_name: Name of the dataset
        subset_name: Subset name
        embedding_type: 'text' or 'image'
        device: Device to use
        cache_dir: Cache directory
        intermediate_layer_names: Layer names for intermediate features
        batch_idx: Batch index for partial caching
        compute_func: Function to compute embeddings if not cached
        
    Returns:
        Computed or loaded embeddings
    """
    # Check if we should use caching
    if not should_use_cache(embedding_model, aligning_model):
        # CLIPMultiLayerFTAlignment case - compute directly
        if compute_func is not None:
            return compute_func(data_batch, embedding_model, device=device, 
                              intermediate_layer_names=intermediate_layer_names)
        else:
            raise ValueError("compute_func must be provided when not using cache")
    
    # Try to load from cache first
    cached_embeddings = load_cached_embeddings(dataset_name, subset_name, embedding_type, device, cache_dir)
    
    if cached_embeddings is not None:
        # TODO: Add logic to extract the correct batch from cached embeddings
        # For now, assume we're caching entire dataset embeddings
        return cached_embeddings
    
    # Compute embeddings if not cached
    if compute_func is not None:
        embeddings = compute_func(data_batch, embedding_model, device=device,
                                intermediate_layer_names=intermediate_layer_names)
        
        # Apply alignment model if provided
        if aligning_model is not None:
            if embedding_type == "image":
                embeddings = aligning_model.encode_image(embeddings)
            else:  # text
                embeddings = aligning_model.encode_text(embeddings)
        else:
            embeddings = embeddings["final"]
        
        # Normalize embeddings
        embeddings = embeddings.float().to(device)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings
    else:
        raise ValueError("compute_func must be provided when embeddings are not cached")


class EmbeddingCache:
    """
    Context manager for handling embedding caching across an entire evaluation.
    """
    
    def __init__(
        self, 
        dataset_name: str,
        subset_name: str,
        embedding_model,
        aligning_model=None,
        device: str = "cuda",
        cache_dir: str = "./cache"
    ):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.embedding_model = embedding_model
        self.aligning_model = aligning_model
        self.device = device
        self.cache_dir = cache_dir
        self.use_cache = should_use_cache(embedding_model, aligning_model)
        
        # Storage for embeddings to save at the end
        self.image_embeddings = []
        self.text_embeddings = []
        self.foil_embeddings = []  # For datasets with foils
        
        # Cached embeddings loaded at start
        self.cached_image_embeddings = None
        self.cached_text_embeddings = None
        self.cached_foil_embeddings = None
        
        if self.use_cache:
            self._load_cached_embeddings()
    
    def _load_cached_embeddings(self):
        """Load all cached embeddings at the start."""
        self.cached_image_embeddings = load_cached_embeddings(
            self.dataset_name, self.subset_name, "image", self.device, self.cache_dir
        )
        self.cached_text_embeddings = load_cached_embeddings(
            self.dataset_name, self.subset_name, "text", self.device, self.cache_dir
        )
        self.cached_foil_embeddings = load_cached_embeddings(
            self.dataset_name, self.subset_name, "foil", self.device, self.cache_dir
        )
    
    def get_or_compute_embeddings(
        self,
        data_batch: List[Any],
        embedding_type: str,
        compute_func: callable,
        intermediate_layer_names: List[str] = ["final"],
        start_idx: int = 0
    ) -> torch.Tensor:
        """
        Get embeddings from cache or compute them.
        
        Args:
            data_batch: Batch of data
            embedding_type: 'image', 'text', or 'foil'
            compute_func: Function to compute embeddings
            intermediate_layer_names: Layer names
            start_idx: Starting index in the dataset for this batch
            
        Returns:
            Embeddings tensor
        """
        if not self.use_cache:
            # Compute directly for CLIPMultiLayerFTAlignment
            return self._compute_and_align(data_batch, embedding_type, compute_func, intermediate_layer_names)
        
        # Check if we have cached CLIP embeddings for this type
        cached = getattr(self, f"cached_{embedding_type}_embeddings")
        if cached is not None:
            # Extract the batch from cached embeddings
            batch_size = len(data_batch) if isinstance(data_batch, (list, tuple)) else data_batch.shape[0]
            end_idx = start_idx + batch_size
            
            # Ensure we don't go out of bounds
            if end_idx > len(cached):
                raise RuntimeError(
                    f"Cache index out of bounds: trying to access [{start_idx}:{end_idx}] "
                    f"but cached embeddings only have {len(cached)} items. "
                    f"This suggests the cache was built with different data or batch sizes."
                )
            
            clip_embeddings = cached[start_idx:end_idx]
            
            # If we have an aligning model, apply it to the cached CLIP embeddings
            if self.aligning_model is not None:
                if embedding_type == "image":
                    # For images, CLIP embeddings are intermediate features dict
                    # We need to reconstruct the dict format expected by aligning model
                    clip_features = {"final": clip_embeddings}
                    aligned_embeddings = self.aligning_model.encode_image(clip_features)
                else:  # text or foil
                    clip_features = {"final": clip_embeddings}
                    aligned_embeddings = self.aligning_model.encode_text(clip_features)
                
                # Normalize aligned embeddings
                aligned_embeddings = aligned_embeddings.float().to(self.device)
                aligned_embeddings = aligned_embeddings / aligned_embeddings.norm(dim=-1, keepdim=True)
                return aligned_embeddings
            else:
                # No alignment model, return cached CLIP embeddings directly
                return clip_embeddings
        
        # Compute and store for later caching
        embeddings = self._compute_and_align(data_batch, embedding_type, compute_func, intermediate_layer_names)
        
        # Only cache CLIP embeddings (before alignment)
        if self.aligning_model is None:
            # Store for saving at the end (only if no alignment model)
            embedding_list = getattr(self, f"{embedding_type}_embeddings")
            embedding_list.append(embeddings.cpu())
        else:
            # Store raw CLIP embeddings for caching, but return aligned embeddings
            raw_clip_embeddings = self._compute_clip_embeddings(data_batch, embedding_type, compute_func, intermediate_layer_names)
            embedding_list = getattr(self, f"{embedding_type}_embeddings")
            embedding_list.append(raw_clip_embeddings.cpu())
        
        return embeddings
    
    def _compute_clip_embeddings(self, data_batch, embedding_type, compute_func, intermediate_layer_names):
        """Compute raw CLIP embeddings (before alignment) for caching."""
        # Compute raw CLIP embeddings
        embeddings = compute_func(
            data_batch, self.embedding_model, device=self.device,
            intermediate_layer_names=intermediate_layer_names
        )
        
        # Get final layer embeddings (no alignment applied)
        clip_embeddings = embeddings["final"]
        
        # Normalize CLIP embeddings
        clip_embeddings = clip_embeddings.float().to(self.device)
        clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=-1, keepdim=True)
        
        return clip_embeddings
    
    def _compute_and_align(self, data_batch, embedding_type, compute_func, intermediate_layer_names):
        """Compute embeddings and apply alignment."""
        # Compute raw embeddings
        embeddings = compute_func(
            data_batch, self.embedding_model, device=self.device,
            intermediate_layer_names=intermediate_layer_names
        )
        
        # Apply alignment model if provided
        if self.aligning_model is not None:
            # FIX: Ensure embeddings are on the same device as the aligning model
            model_device = next(self.aligning_model.parameters()).device
            
            if isinstance(embeddings, dict):
                # Handle dict of embeddings (multi-layer case)
                embeddings = {
                    k: v.to(model_device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in embeddings.items()
                }
            elif isinstance(embeddings, (list, tuple)):
                # Handle list of embeddings
                embeddings = [
                    emb.to(model_device, non_blocking=True) if isinstance(emb, torch.Tensor) else emb 
                    for emb in embeddings
                ]
            elif isinstance(embeddings, torch.Tensor):
                # Handle single tensor
                embeddings = embeddings.to(model_device, non_blocking=True)
            
            # Now apply alignment
            if embedding_type == 'image':
                embeddings = self.aligning_model.encode_image(embeddings)
            else:  # text
                embeddings = self.aligning_model.encode_text(embeddings)
        else:
            embeddings = embeddings["final"]
            # Normalize final embeddings
            embeddings = embeddings.float().to(self.device)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def compute_tqa_binary_similarity(
        self,
        images: torch.Tensor,
        pos_texts: List[str],
        neg_texts: List[str],
        compute_image_func: callable,
        compute_text_func: callable,
        intermediate_image_layer_names: List[str] = ["final"],
        intermediate_text_layer_names: List[str] = ["final"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute binary choice similarities.
        
        Args:
            images: Image batch [B, C, H, W]
            pos_texts: Positive caption strings
            neg_texts: Negative caption strings
            compute_image_func: Function to compute image embeddings
            compute_text_func: Function to compute text embeddings
            intermediate_image_layer_names: Layer names for image features
            intermediate_text_layer_names: Layer names for text features
            
        Returns:
            Tuple of (pos_scores, neg_scores, img_embs, pos_embs, neg_embs)
            - pos_scores: [B] similarity with positive captions
            - neg_scores: [B] similarity with negative captions
            - img_embs: [B, D] image embeddings
            - pos_embs: [B, D] positive text embeddings
            - neg_embs: [B, D] negative text embeddings
        """
        # Standard CLIP path: compute embeddings separately
        img_embs = self.get_or_compute_embeddings(
            images, "image", compute_image_func, 
            intermediate_image_layer_names
        )
        pos_embs = self.get_or_compute_embeddings(
            pos_texts, "text", compute_text_func,
            intermediate_text_layer_names
        )
        neg_embs = self.get_or_compute_embeddings(
            neg_texts, "foil", compute_text_func,
            intermediate_text_layer_names
        )
        
        # Compute scores
        pos_scores = (img_embs * pos_embs).sum(dim=-1)
        neg_scores = (img_embs * neg_embs).sum(dim=-1)
        
        return pos_scores, neg_scores, img_embs.cpu(), pos_embs.cpu(), neg_embs.cpu()
    
    def compute_tqa_multichoice_similarity(
        self,
        images: torch.Tensor,
        caption_options: List[List[str]],
        compute_image_func: callable,
        compute_text_func: callable,
        intermediate_image_layer_names: List[str] = ["final"],
        intermediate_text_layer_names: List[str] = ["final"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute similarities for multi-choice tasks.
        
        Args:
            images: Image batch [B, C, H, W]
            caption_options: List of caption option lists [[cap1, cap2, ...], ...]
            compute_image_func: Function to compute image embeddings
            compute_text_func: Function to compute text embeddings
            intermediate_image_layer_names: Layer names for image features
            intermediate_text_layer_names: Layer names for text features
            
        Returns:
            Tuple of (similarity_matrix, img_embs, txt_embs)
            - similarity_matrix: [B, num_options] similarity scores
            - img_embs: [B, D] image embeddings
            - txt_embs: [B, num_options, D] text embeddings
        """
        B = len(caption_options)
        num_options = len(caption_options[0])
        
        # Flatten captions
        flat_captions = []
        for options in caption_options:
            flat_captions.extend(options)
        
        # Standard CLIP path
        img_embs = self.get_or_compute_embeddings(
            images, "image", compute_image_func,
            intermediate_image_layer_names
        )
        txt_embs_flat = self.get_or_compute_embeddings(
            flat_captions, "text", compute_text_func,
            intermediate_text_layer_names
        )
        
        # Reshape text embeddings
        D = txt_embs_flat.shape[-1]
        txt_embs = txt_embs_flat.view(B, num_options, D)
        
        # Compute similarity matrix: [B, num_options]
        img_embs_exp = img_embs.unsqueeze(1)  # [B, 1, D]
        similarity_matrix = (img_embs_exp * txt_embs).sum(dim=-1)  # [B, num_options]
        
        return similarity_matrix, img_embs.cpu(), txt_embs.cpu()
    
    def compute_tqa_colorswap_similarity(
        self,
        images_1: torch.Tensor,
        images_2: torch.Tensor,
        captions_1: List[str],
        captions_2: List[str],
        compute_image_func: callable,
        compute_text_func: callable,
        intermediate_image_layer_names: List[str] = ["final"],
        intermediate_text_layer_names: List[str] = ["final"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute similarities for ColorSwap task.
        
        ColorSwap has 2 images and 2 captions per sample:
        - Correct matching: image_1 <-> caption_1, image_2 <-> caption_2
        - Text score: Does caption_1 prefer image_1 over image_2?
        - Image score: Does image_1 prefer caption_1 over caption_2?
        
        Args:
            images_1: First image batch [B, C, H, W]
            images_2: Second image batch [B, C, H, W]
            captions_1: First caption list
            captions_2: Second caption list
            compute_image_func: Function to compute image embeddings
            compute_text_func: Function to compute text embeddings
            intermediate_image_layer_names: Layer names for image features
            intermediate_text_layer_names: Layer names for text features
            
        Returns:
            Tuple of (text_correct, text_incorrect, image_correct, image_incorrect, 
                      img_embs_1, img_embs_2, cap_embs_1, cap_embs_2)
        """
        B = len(captions_1)
        
        # Standard CLIP path
        # Combine images for efficient batch processing
        combined_images = torch.cat([images_1, images_2], dim=0)  # [2B, C, H, W]
        combined_img_embs = self.get_or_compute_embeddings(
            combined_images, "image", compute_image_func,
            intermediate_image_layer_names
        )
        img_embs_1 = combined_img_embs[:B]
        img_embs_2 = combined_img_embs[B:]
        
        # Combine captions for efficient batch processing
        all_captions = captions_1 + captions_2
        all_cap_embs = self.get_or_compute_embeddings(
            all_captions, "text", compute_text_func,
            intermediate_text_layer_names
        )
        cap_embs_1 = all_cap_embs[:B]
        cap_embs_2 = all_cap_embs[B:]
        
        # Text scores: caption_1 prefers image_1 over image_2
        text_correct = (cap_embs_1 * img_embs_1).sum(dim=-1)
        text_incorrect = (cap_embs_1 * img_embs_2).sum(dim=-1)
        
        # Image scores: image_1 prefers caption_1 over caption_2
        image_correct = (img_embs_1 * cap_embs_1).sum(dim=-1)
        image_incorrect = (img_embs_1 * cap_embs_2).sum(dim=-1)
        
        return (text_correct, text_incorrect, image_correct, image_incorrect,
                img_embs_1.cpu(), img_embs_2.cpu(), cap_embs_1.cpu(), cap_embs_2.cpu())
    
    def save_embeddings(self):
        """Save all computed embeddings to cache."""
        if not self.use_cache :
            return
        
        if self.aligning_model is not None:
            return
        
        for embedding_type in ["image", "text", "foil"]:
            embedding_list = getattr(self, f"{embedding_type}_embeddings")
            if embedding_list:
                # Concatenate all batches
                all_embeddings = torch.cat(embedding_list, dim=0)
                save_cached_embeddings(
                    all_embeddings, self.dataset_name, self.subset_name, 
                    embedding_type, self.cache_dir
                )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_embeddings()
