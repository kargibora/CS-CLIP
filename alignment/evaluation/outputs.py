"""Functions for gathering validation outputs from batches."""

from typing import Any, Dict, Union

import torch

from .utils import move_to_device, get_base_model


def compute_embeddings_and_negatives(
    model: Any,
    neg_captions_all: torch.Tensor,
    emb_dim: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute negative text embeddings for all negatives in the batch.
    
    Args:
        model: Model with encode_text method
        neg_captions_all: Negative captions [B, K, seq_len]
        emb_dim: Embedding dimension
        device: Device for computations
        
    Returns:
        Negative text embeddings [B, K, D]
    """
    B, K, seq_len = neg_captions_all.shape
    neg_flat = neg_captions_all.view(B*K, seq_len)
    with torch.no_grad():
        neg_emb_flat = model.encode_text(neg_flat)
        neg_emb_flat = neg_emb_flat.to(device)
    neg_text_embeddings_all = neg_emb_flat.view(B, K, -1)
    return neg_text_embeddings_all


def _unpack_batch_finetune(batch: Union[Dict, tuple]) -> tuple:
    """
    Unpack fine-tuning batch format.
    
    Handles both:
    - NEW FORMAT (paired negatives): neg_tokens is [B, 1+N, 77] paired with pos_tokens
    - LEGACY FORMAT: neg_token is [B, 77] single negative, all_neg_tokens is [B, 3, 77]
    """
    if isinstance(batch, dict):
        images = batch["images"]
        captions = batch["pos_tokens"]
        
        # Check for NEW format (paired negatives) vs LEGACY format
        if "neg_tokens" in batch:
            # NEW FORMAT: neg_tokens is [B, 1+N, 77] - paired with pos_tokens
            neg_tokens_all = batch["neg_tokens"]  # [B, 1+N, 77]
            
            # For evaluation, we use the full caption negative (index 0)
            # and use all paired negatives for computing metrics
            if neg_tokens_all.ndim == 3 and neg_tokens_all.shape[1] > 1:
                # Paired format: [B, 1+N, 77]
                neg_captions = neg_tokens_all[:, 0, :]  # [B, 77] - negative for full caption
                neg_captions_all = neg_tokens_all  # [B, 1+N, 77] - all paired negatives
            else:
                # Single negative: [B, 1, 77] or [B, 77]
                neg_captions = neg_tokens_all.squeeze(1) if neg_tokens_all.ndim == 3 else neg_tokens_all
                neg_captions_all = neg_tokens_all.unsqueeze(1) if neg_tokens_all.ndim == 2 else neg_tokens_all
        else:
            # LEGACY FORMAT: separate neg_token and all_neg_tokens
            neg_captions = batch.get("neg_token")
            neg_captions_all = batch.get("all_neg_tokens", neg_captions)
            if neg_captions_all is not None and neg_captions_all.ndim == 2:
                neg_captions_all = neg_captions_all.unsqueeze(1)
    else:
        if len(batch) == 4:
            images, captions, neg_captions, neg_captions_all = batch
        else:
            images, captions, neg_captions = batch
            neg_captions_all = neg_captions
    
    return images, captions, neg_captions, neg_captions_all


def _unpack_batch_cached(batch: Union[Dict, tuple]) -> tuple:
    """Unpack cached embeddings batch format."""
    if len(batch) == 4:
        image_emb, text_emb, neg_emb, neg_emb_all = batch
    else:
        image_emb, text_emb, neg_emb = batch
        neg_emb_all = neg_emb
    
    return image_emb, text_emb, neg_emb, neg_emb_all


def _encode_all_negatives_dict(
    model: Any,
    neg_emb_all: Dict[str, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Encode all negatives when input is dict format.
    
    Args:
        model: Model with encode_text method
        neg_emb_all: Dict of negative embeddings {layer: [B, K, D_l]}
        device: Device for computations
        
    Returns:
        Encoded negatives [B, K, D]
    """
    # Extract the main embedding tensor
    if 'final' in neg_emb_all:
        neg_emb_all_tensor = neg_emb_all['final']
    else:
        neg_emb_all_tensor = list(neg_emb_all.values())[-1]
    
    neg_txt_emb_all = torch.zeros_like(neg_emb_all_tensor, dtype=torch.float32, device=device)
    
    for i in range(neg_emb_all_tensor.shape[1]):
        # Reconstruct dict format for each negative
        negs_dict = {k: v[:, i, :] for k, v in neg_emb_all.items()}
        neg_emb_result = model.encode_text(negs_dict)
        if torch.is_tensor(neg_emb_result):
            neg_txt_emb_all[:, i, :] = neg_emb_result.to(device)
    
    return neg_txt_emb_all


def _encode_all_negatives_tensor(
    model: Any,
    neg_emb_all: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Encode all negatives when input is tensor format.
    
    Args:
        model: Model with encode_text method
        neg_emb_all: Negative embeddings [B, K, D]
        device: Device for computations
        
    Returns:
        Encoded negatives [B, K, D]
    """
    neg_txt_emb_all = torch.zeros_like(neg_emb_all, dtype=torch.float32, device=device)
    
    for i in range(neg_emb_all.shape[1]):
        negs = neg_emb_all[:, i, :]
        neg_emb_result = model.encode_text(negs)
        if torch.is_tensor(neg_emb_result):
            neg_txt_emb_all[:, i, :] = neg_emb_result.to(device)
    
    return neg_txt_emb_all


def gather_validation_outputs(
    model: Any,
    batch: Union[Dict, tuple],
    device: torch.device,
    is_finetune: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Unpacks and encodes images/captions/negatives depending on dataloader output format.
    
    Args:
        model: Model for encoding
        batch: Batch from dataloader (dict or tuple)
        device: Device for computations
        is_finetune: Whether this is fine-tuning mode
        
    Returns:
        Dictionary with image_embeddings, pos_text_embeddings, neg_text_embeddings, neg_text_embeddings_all
    """
    if is_finetune:
        images, captions, neg_captions, neg_captions_all = _unpack_batch_finetune(batch)
        
        images, captions, neg_captions, neg_captions_all = [
            x.to(device) for x in (images, captions, neg_captions, neg_captions_all)
        ]
        
        if neg_captions_all.ndim == 2:
            neg_captions_all = neg_captions_all.unsqueeze(1)

        img_emb = model.encode_image(images)
        
        # Handle multi-caption mode: captions might be [B, 1+N, 77]
        if captions.ndim == 3:
            # Multi-caption mode: reshape to [B*(1+N), 77], encode, then reshape back
            B, N, seq_len = captions.shape
            captions_flat = captions.reshape(B * N, seq_len)
            pos_txt_emb_flat = model.encode_text(captions_flat)  # [B*(1+N), D]
            pos_txt_emb = pos_txt_emb_flat.reshape(B, N, -1)  # [B, 1+N, D]
        else:
            # Standard mode: [B, 77]
            pos_txt_emb = model.encode_text(captions)
        
        # Handle negatives similarly
        if neg_captions.ndim == 3:
            B, N, seq_len = neg_captions.shape
            neg_captions_flat = neg_captions.reshape(B * N, seq_len)
            neg_txt_emb_flat = model.encode_text(neg_captions_flat)
            neg_txt_emb = neg_txt_emb_flat.reshape(B, N, -1)
        else:
            neg_txt_emb = model.encode_text(neg_captions)
        
        # Ensure embeddings are on the correct device
        img_emb = img_emb.to(device)
        pos_txt_emb = pos_txt_emb.to(device)
        neg_txt_emb = neg_txt_emb.to(device)
        
        neg_txt_emb_all = compute_embeddings_and_negatives(
            model, neg_captions_all, pos_txt_emb.shape[-1], device
        )
        
        return dict(
            image_embeddings=img_emb,
            pos_text_embeddings=pos_txt_emb,
            neg_text_embeddings=neg_txt_emb,
            neg_text_embeddings_all=neg_txt_emb_all
        )
    
    else:
        image_emb, text_emb, neg_emb, neg_emb_all = _unpack_batch_cached(batch)
        
        # Move to device
        image_emb, text_emb, neg_emb, neg_emb_all = [
            move_to_device(x, device) for x in (image_emb, text_emb, neg_emb, neg_emb_all)
        ]
        
        # Check if neg_emb_all needs unsqueezing
        if torch.is_tensor(neg_emb_all) and neg_emb_all.ndim == 2:
            neg_emb_all = neg_emb_all.unsqueeze(1)
        elif isinstance(neg_emb_all, dict):
            for k, v in neg_emb_all.items():
                if torch.is_tensor(v) and v.ndim == 2:
                    neg_emb_all[k] = v.unsqueeze(1)
        
        # Encode
        img_emb = model.encode_image(image_emb)
        
        # Handle multi-caption mode for cached embeddings too
        if torch.is_tensor(text_emb) and text_emb.ndim == 3:
            B, N, seq_len = text_emb.shape
            text_emb_flat = text_emb.reshape(B * N, seq_len)
            pos_txt_emb_flat = model.encode_text(text_emb_flat)
            pos_txt_emb = pos_txt_emb_flat.reshape(B, N, -1)
        else:
            pos_txt_emb = model.encode_text(text_emb)
        
        if torch.is_tensor(neg_emb) and neg_emb.ndim == 3:
            B, N, seq_len = neg_emb.shape
            neg_emb_flat = neg_emb.reshape(B * N, seq_len)
            neg_txt_emb_flat = model.encode_text(neg_emb_flat)
            neg_txt_emb = neg_txt_emb_flat.reshape(B, N, -1)
        else:
            neg_txt_emb = model.encode_text(neg_emb)
        
        # Ensure embeddings are on the correct device
        if torch.is_tensor(img_emb):
            img_emb = img_emb.to(device)
        if torch.is_tensor(pos_txt_emb):
            pos_txt_emb = pos_txt_emb.to(device)
        if torch.is_tensor(neg_txt_emb):
            neg_txt_emb = neg_txt_emb.to(device)
        
        # Handle neg_emb_all - could be tensor or dict
        if isinstance(neg_emb_all, dict):
            neg_txt_emb_all = _encode_all_negatives_dict(model, neg_emb_all, device)
        else:
            neg_txt_emb_all = _encode_all_negatives_tensor(model, neg_emb_all, device)
        
        return dict(
            image_embeddings=img_emb,
            pos_text_embeddings=pos_txt_emb,
            neg_text_embeddings=neg_txt_emb,
            neg_text_embeddings_all=neg_txt_emb_all
        )


def gather_validation_outputs_multilayer(
    model: Any,
    batch: tuple,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    For multi-layer models with dict-of-tensors format.
    
    Args:
        model: Multi-layer model
        batch: Batch of embedding dicts
        device: Device for computations
        
    Returns:
        Dictionary with image_embeddings, pos_text_embeddings, neg_text_embeddings, neg_text_embeddings_all
    """
    image_emb_dict, text_emb_dict, neg_text_emb_dict, all_neg_text_emb_dict = batch

    # Get actual device from model
    model_device = next(model.parameters()).device
    if model_device != device:
        device = model_device
    
    # Move each dict's tensors to device
    image_emb_dict = {k: v.to(device) for k, v in image_emb_dict.items()}
    text_emb_dict = {k: v.to(device) for k, v in text_emb_dict.items()}
    neg_text_emb_dict = {k: v.to(device) for k, v in neg_text_emb_dict.items()}
    all_neg_text_emb_dict = {k: v.to(device) for k, v in all_neg_text_emb_dict.items()}

    model = model.to(device)

    # Encode embeddings
    img_emb = model.encode_image(image_emb_dict)
    pos_txt_emb = model.encode_text(text_emb_dict)
    neg_txt_emb = model.encode_text(neg_text_emb_dict)

    # For all negatives: shape (batch, K, D)
    K = next(iter(all_neg_text_emb_dict.values())).shape[1]
    batch_size = next(iter(all_neg_text_emb_dict.values())).shape[0]
    emb_dim = img_emb.shape[-1]

    neg_txt_emb_all = torch.zeros((batch_size, K, emb_dim), dtype=img_emb.dtype, device=img_emb.device)
    
    for k in range(K):
        neg_dict_k = {ln: v[:, k, ...] for ln, v in all_neg_text_emb_dict.items()}
        neg_emb_k = model.encode_text(neg_dict_k)
        neg_txt_emb_all[:, k, :] = neg_emb_k

    return dict(
        image_embeddings=img_emb,
        pos_text_embeddings=pos_txt_emb,
        neg_text_embeddings=neg_txt_emb,
        neg_text_embeddings_all=neg_txt_emb_all
    )

