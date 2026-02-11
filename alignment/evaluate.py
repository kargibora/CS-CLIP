"""
Evaluation functions for CLIP models.

This module is now a thin wrapper around the refactored evaluation package.
All implementation has been moved to alignment/evaluation/ for better organization.

For direct imports, use:
    from alignment.evaluation import evaluate, evaluate_labclip, evaluate_ft
"""

# Re-export all main functions from the new package for backwards compatibility
from alignment.evaluation import (
    evaluate,
    evaluate_ft,
    MetricKeys,
    ModelType,
    is_labclip_model,
    is_ft_mode,
)

from utils.evaluate import (
    get_contrastive_accuracy,
    get_negative_similarity,
    get_negative_similarity_img,
    get_caption_image_similarity,
)
__all__ = [
    'evaluate',
    'evaluate_labclip',
    'evaluate_ft',
    'MetricKeys',
    'ModelType',
    'is_labclip_model',
    'is_ft_mode',
]

from typing import Any, Callable, Dict
import torch
import torch.distributed as dist
import logging
from tqdm import tqdm


def ddp_sum_scalar_metrics(metrics_sums: Dict[str, float],
                           unified_keys: list,
                           num_batches: int,
                           num_samples_local: int,
                           device: torch.device):
    """
    Reduce scalar metrics across ranks using all_reduce (sum).
    Assumes keys are already synced and padded.
    """
    # If no metrics were computed, just sync the counts.
    if not unified_keys:
        buffer = torch.tensor([float(num_batches), float(num_samples_local)], dtype=torch.float32, device=device)
        # logging.info("No metrics to sum, only synchronizing counts.")
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        # logging.info(f"Total batches: {int(buffer[0].item())}, Global samples: {int(buffer[1].item())}")
        total_batches = int(buffer[0].item())
        num_samples_global = int(buffer[1].item())
        return {}, total_batches, num_samples_global

    # Ensure all keys exist in the local dict before creating the tensor,
    # even if a rank processed 0 batches. This guarantees tensor size consistency.
    for k in unified_keys:
        metrics_sums.setdefault(k, 0.0)

    # Order: num_batches, num_samples, then all metrics_sums alphabetically
    values = [float(num_batches), float(num_samples_local)] + [metrics_sums[k] for k in unified_keys]
    buffer = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Sum across all ranks
    # logging.info(f"Rank {dist.get_rank()}: Summing metrics: {values}")
    # logging.info(f"Rank {dist.get_rank()}: Buffer device: {buffer.device}, Buffer shape: {buffer.shape}")
    
    # Ensure buffer is on the correct device for this rank
    if torch.cuda.is_available() and device.type == 'cuda':
        current_device = torch.cuda.current_device()
        if buffer.device.index != current_device:
            logging.warning(f"Rank {dist.get_rank()}: Moving buffer from {buffer.device} to cuda:{current_device}")
            buffer = buffer.to(f'cuda:{current_device}')
    
    # logging.info(f"Rank {dist.get_rank()}: About to call all_reduce...")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    # logging.info(f"Rank {dist.get_rank()}: all_reduce completed successfully")
    
    # Unpack summed values
    total_batches = int(buffer[0].item())
    num_samples_global = int(buffer[1].item())
    
    # Reconstruct the metrics dict
    aggregated_metrics = {k: buffer[i+2].item() for i, k in enumerate(unified_keys)}
    
    return aggregated_metrics, total_batches, num_samples_global


def ddp_sum_tensor_metrics(per_neg_sums: Dict[str, torch.Tensor],
                           unified_keys: list,
                           device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Reduce per-neg tensors using all_reduce. 
    Assumes keys and shapes are already synced and padded.
    """
    if not unified_keys:
        return {}

    # All-reduce each tensor. Since they are pre-padded, this is safe.
    for k in unified_keys:
        tensor = per_neg_sums[k].to(device)
        
        # Ensure tensor is on the correct device for this rank
        if torch.cuda.is_available() and device.type == 'cuda':
            current_device = torch.cuda.current_device()
            if tensor.device.index != current_device:
                logging.warning(f"Rank {dist.get_rank()}: Moving tensor {k} from {tensor.device} to cuda:{current_device}")
                tensor = tensor.to(f'cuda:{current_device}')
        
        # logging.info(f"Rank {dist.get_rank()}: About to all_reduce tensor {k} with shape {tensor.shape}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # logging.info(f"Rank {dist.get_rank()}: all_reduce completed for tensor {k}")
        per_neg_sums[k] = tensor
    
    return per_neg_sums


def compute_embeddings_and_negatives(
    model: Any,
    neg_captions_all: torch.Tensor,   # [B, K, seq_len]
    emb_dim: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute negative text embeddings for all negatives in the batch. Returns: [B, K, D]
    """
    B, K, seq_len = neg_captions_all.shape
    neg_flat = neg_captions_all.view(B*K, seq_len)      # [B*K, seq_len]
    with torch.no_grad():
        neg_emb_flat = model.encode_text(neg_flat)       # [B*K, D]
        neg_emb_flat = neg_emb_flat.to(device)          # Ensure correct device
    neg_text_embeddings_all = neg_emb_flat.view(B, K, -1)  # [B, K, D]
    return neg_text_embeddings_all

def gather_validation_outputs(
    model: Any,
    batch: tuple,
    device: torch.device,
    is_finetune: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Unpacks and encodes images/captions/negatives depending on dataloader output format.
    Returns a dict of embeddings.
    
    Handles both:
    - NEW FORMAT (paired negatives): neg_tokens is [B, 1+N, 77] paired with pos_tokens
    - LEGACY FORMAT: neg_token is [B, 77] single negative, all_neg_tokens is [B, 3, 77]
    """
    if is_finetune:
        # Handle both dictionary format (from collate_fn) and tuple format (legacy)
        if isinstance(batch, dict):
            # Dictionary format from collate_fn
            images = batch["images"]
            captions = batch["pos_tokens"]
            
            # Check for NEW format (paired negatives) vs LEGACY format
            if "neg_tokens" in batch:
                # NEW FORMAT: neg_tokens is [B, 1+N, 77] - paired with pos_tokens
                neg_tokens_all = batch["neg_tokens"]  # [B, 1+N, 77]
                
                # For evaluation, we use the full caption negative (index 0)
                # and can use all paired negatives as additional negatives
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
                neg_captions = batch.get("neg_token", batch.get("neg_tokens"))
                neg_captions_all = batch.get("all_neg_tokens", neg_captions)
                if neg_captions_all.ndim == 2:
                    neg_captions_all = neg_captions_all.unsqueeze(1)
        else:
            # Tuple format (legacy)
            if len(batch) == 4:
                images, captions, neg_captions, neg_captions_all = batch
            else:
                images, captions, neg_captions = batch
                neg_captions_all = neg_captions
        
        images, captions, neg_captions, neg_captions_all = [
            x.to(device) for x in (images, captions, neg_captions, neg_captions_all)
        ]
        if neg_captions_all.ndim == 2:
            neg_captions_all = neg_captions_all.unsqueeze(1)
        
        # Handle multi-caption mode: captions might be [B, 1+N, seq_len]
        # For evaluation, we only use the first (full) caption
        if captions.ndim == 3:
            # Multi-caption mode: [B, 1+N, seq_len] -> [B, seq_len]
            captions = captions[:, 0, :]
        if neg_captions.ndim == 3:
            # Multi-caption mode: [B, 1+N, seq_len] -> [B, seq_len]
            neg_captions = neg_captions[:, 0, :]

        img_emb = model.encode_image(images)
        pos_txt_emb = model.encode_text(captions)
        neg_txt_emb = model.encode_text(neg_captions)
        
        # Ensure embeddings are on the correct device
        img_emb = img_emb.to(device)
        pos_txt_emb = pos_txt_emb.to(device)
        neg_txt_emb = neg_txt_emb.to(device)
        
        neg_txt_emb_all = compute_embeddings_and_negatives(model, neg_captions_all, pos_txt_emb.shape[-1], device)
        return dict(
            image_embeddings=img_emb,
            pos_text_embeddings=pos_txt_emb,
            neg_text_embeddings=neg_txt_emb,
            neg_text_embeddings_all=neg_txt_emb_all
        )
    else:
        if len(batch) == 4:
            image_emb, text_emb, neg_emb, neg_emb_all = batch
        else:
            image_emb, text_emb, neg_emb = batch
            neg_emb_all = neg_emb
        
        # Handle dict or tensor inputs - move to device appropriately
        def move_to_device(x, device):
            if isinstance(x, dict):
                return {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
            elif torch.is_tensor(x):
                return x.to(device)
            return x
        
        image_emb, text_emb, neg_emb, neg_emb_all = [
            move_to_device(x, device) for x in (image_emb, text_emb, neg_emb, neg_emb_all)
        ]
        
        # Check if neg_emb_all needs unsqueezing (only for tensors)
        if torch.is_tensor(neg_emb_all) and neg_emb_all.ndim == 2:
            neg_emb_all = neg_emb_all.unsqueeze(1)
        elif isinstance(neg_emb_all, dict):
            # For dict format, check each value
            for k, v in neg_emb_all.items():
                if torch.is_tensor(v) and v.ndim == 2:
                    neg_emb_all[k] = v.unsqueeze(1)
        
        img_emb = model.encode_image(image_emb)
        pos_txt_emb = model.encode_text(text_emb)
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
            # For dict format, extract the main embedding tensor
            # Assume dict has structure like {'layer_12': tensor, ...}
            # Use the last/final layer or a specific key
            if 'final' in neg_emb_all:
                neg_emb_all_tensor = neg_emb_all['final']
            else:
                # Get the highest layer number or last key
                neg_emb_all_tensor = list(neg_emb_all.values())[-1]
            
            neg_txt_emb_all = torch.zeros_like(neg_emb_all_tensor, dtype=torch.float32, device=device)
            for i in range(neg_emb_all_tensor.shape[1]):
                # Reconstruct dict format for each negative
                negs_dict = {k: v[:, i, :] for k, v in neg_emb_all.items()}
                neg_emb_result = model.encode_text(negs_dict)
                if torch.is_tensor(neg_emb_result):
                    neg_txt_emb_all[:, i, :] = neg_emb_result.to(device)
        else:
            # Tensor format
            neg_txt_emb_all = torch.zeros_like(neg_emb_all, dtype=torch.float32, device=device)
            for i in range(neg_emb_all.shape[1]):
                negs = neg_emb_all[:, i, :]
                neg_emb_result = model.encode_text(negs)
                if torch.is_tensor(neg_emb_result):
                    neg_txt_emb_all[:, i, :] = neg_emb_result.to(device)
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
    For new multi-layer models and dataset:
    - Moves all dict-of-tensors to device
    - Computes aligned embeddings for images and captions
    - Returns output dict with same keys as before
    """
    image_emb_dict, text_emb_dict, neg_text_emb_dict, all_neg_text_emb_dict = batch

    # Ensure model is on the correct device and get the actual device from model
    model_device = next(model.parameters()).device
    if model_device != device:
        # If model is on a different device than expected, use the model's device
        device = model_device
    
    # Move each dict's tensors to the model's device
    image_emb_dict      = {k: v.to(device) for k, v in image_emb_dict.items()}
    text_emb_dict       = {k: v.to(device) for k, v in text_emb_dict.items()}
    neg_text_emb_dict   = {k: v.to(device) for k, v in neg_text_emb_dict.items()}
    all_neg_text_emb_dict = {k: v.to(device) for k, v in all_neg_text_emb_dict.items()}

    # Ensure model is definitely on the correct device
    model = model.to(device)

    # Encode (align+fuse) embeddings
    img_emb     = model.encode_image(image_emb_dict)         # (batch, D)
    pos_txt_emb = model.encode_text(text_emb_dict)           # (batch, D)
    neg_txt_emb = model.encode_text(neg_text_emb_dict)       # (batch, D)

    # For all negatives: shape (batch, K, D)
    # Need to apply model.encode_text to each K negative (per sample)
    # Let's build (batch, K, D) by iterating over K
    # Assume all_neg_text_emb_dict[ln].shape = (batch, K, D_l)
    K = next(iter(all_neg_text_emb_dict.values())).shape[1]
    batch_size = next(iter(all_neg_text_emb_dict.values())).shape[0]
    emb_dim = img_emb.shape[-1]

    neg_txt_emb_all = torch.zeros((batch_size, K, emb_dim), dtype=img_emb.dtype, device=img_emb.device)
    # For each K, gather and fuse negatives per sample
    for k in range(K):
        neg_dict_k = {ln: v[:, k, ...] for ln, v in all_neg_text_emb_dict.items()}  # {ln: (batch, D_l)}
        neg_emb_k = model.encode_text(neg_dict_k)  # (batch, D)
        neg_txt_emb_all[:, k, :] = neg_emb_k

    return dict(
        image_embeddings=img_emb,                 # (batch, D)
        pos_text_embeddings=pos_txt_emb,          # (batch, D)
        neg_text_embeddings=neg_txt_emb,          # (batch, D)
        neg_text_embeddings_all=neg_txt_emb_all   # (batch, K, D)
    )


def compute_all_metrics(
    img_emb: torch.Tensor,
    pos_txt_emb: torch.Tensor,
    neg_txt_emb: torch.Tensor,
    neg_txt_emb_all: torch.Tensor,
    loss_fn: Callable,
    model: Any,
    device: torch.device,
    **loss_kwargs
) -> Dict[str, Any]:
    """
    Computes all desired metrics for a batch and returns a dict.
    Supports dictionary loss outputs.
    """
    # Ensure all tensors are on the same device
    img_emb = img_emb.to(device)
    pos_txt_emb = pos_txt_emb.to(device)
    neg_txt_emb = neg_txt_emb.to(device)
    neg_txt_emb_all = neg_txt_emb_all.to(device)
    
    metrics = {}
    # Loss and "accuracy" from loss function (returns a dict)
    loss_dict = loss_fn(
        img_emb.squeeze(1) if img_emb.dim() == 3 else img_emb,
        pos_text_embeddings=pos_txt_emb,
        neg_text_embeddings=neg_txt_emb,
        temperature=model.temperature,
        device=device,
        **loss_kwargs
    )
    
    # Merge all loss dict keys in
    for k, v in loss_dict.items():
        metrics[f"val_{k}"] = v if not torch.is_tensor(v) else v.detach()
    
    # Extra metrics as before
    contrastive_acc, n_pos_gt_neg, per_neg_acc = get_contrastive_accuracy(img_emb, pos_txt_emb, neg_txt_emb_all, get_average=False)
    neg_text_sim, per_neg_text_sim = get_negative_similarity(pos_txt_emb, neg_txt_emb_all, get_average=False)
    neg_text_pos_image_sim, per_neg_text_pos_image_sim = get_negative_similarity_img(img_emb, neg_txt_emb_all, get_average=False)
    pos_sim = get_caption_image_similarity(pos_txt_emb, img_emb, get_average=False)

    metrics['contrastive_accuracy'] = contrastive_acc
    metrics['pos_greater_than_neg'] = n_pos_gt_neg
    metrics['neg_text_similarity'] = neg_text_sim
    metrics['neg_text_pos_image_similarity'] = neg_text_pos_image_sim
    metrics['pos_similarity'] = pos_sim
    metrics['per_neg_accuracy'] = per_neg_acc
    metrics['per_neg_text_similarity'] = per_neg_text_sim
    metrics['per_neg_text_pos_image_similarity'] = per_neg_text_pos_image_sim
    metrics['pos_neg_similarity_gap'] = pos_sim - neg_text_pos_image_sim

    return metrics

# === Evaluation Loop ===

def evaluate(
    model: Any,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Callable,
    is_finetune: bool = False,
    **loss_kwargs
) -> Dict[str, Any]:
    # Check if this is a LabCLIP model and route to appropriate evaluation
    model_class_name = model.__class__.__name__
    
    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        actual_model = model.module
        actual_model_class = actual_model.__class__.__name__
    else:
        actual_model = model
        actual_model_class = model_class_name
    
    
    # Original evaluation pipeline for standard CLIP models
    logging.info(f"Using standard evaluation pipeline for {actual_model_class}")
    
    model.eval()
    distributed = dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0

    # Proactive device management for distributed training
    if distributed and torch.cuda.is_available():
        current_cuda_device = torch.cuda.current_device()
        expected_device = f'cuda:{rank}'
        
        # Ensure the device parameter matches the current rank's expected device
        if device.type == 'cuda' and device.index != current_cuda_device:
            # logging.warning(f"Rank {rank}: Device mismatch detected. Expected {expected_device}, got {device}. Correcting...")
            device = torch.device(f'cuda:{current_cuda_device}')
        
        # Set the current device explicitly for this rank
        torch.cuda.set_device(current_cuda_device)
        # logging.info(f"Rank {rank}: Using device {device} (current CUDA device: {current_cuda_device})")

    metrics_sums: Dict[str, float] = {}
    per_neg_sums: Dict[str, torch.Tensor] = None
    num_batches = 0
    num_samples_local = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", disable=(rank != 0))):
            try:
                if model.__class__.__name__ == "FlexibleCLIPMultiLayerAlignment":
                    outputs = gather_validation_outputs_multilayer(model, batch, device)
                else:
                    outputs = gather_validation_outputs(model, batch, device, is_finetune)

                img_emb = outputs['image_embeddings'].float()
                pos_txt_emb = outputs['pos_text_embeddings'].float()
                neg_txt_emb = outputs['neg_text_embeddings'].float()
                neg_txt_emb_all = outputs['neg_text_embeddings_all'].float()

                batch_size = img_emb.size(0)
                num_samples_local += batch_size

                batch_metrics = compute_all_metrics(
                    img_emb, pos_txt_emb, neg_txt_emb, neg_txt_emb_all, loss_fn, model, device, **loss_kwargs
                )

                # Sum scalar metrics locally
                for k, v in batch_metrics.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 0:
                        metrics_sums[k] = metrics_sums.get(k, 0.0) + float(v.item())

                # Sum per-neg tensors locally
                if per_neg_sums is None:
                    per_neg_sums = {k: batch_metrics[k].clone() for k in batch_metrics if 'per_neg' in k}
                else:
                    for k in per_neg_sums:
                        per_neg_sums[k] += batch_metrics[k]

                num_batches += 1

            except Exception as e:
                logging.error(f"Rank {rank}: Error processing batch {batch_idx}: {e}")
                continue

    # logging.info(f"Rank {rank}: Finished processing {num_batches} batches, {num_samples_local} samples")

    # ---- Distributed aggregation (NO gather; only all_reduce) -------------------
    if distributed:
        # logging.info(f"Rank {rank}: Starting distributed aggregation across {world_size} processes")
        dist.barrier()

        # --- 1. Synchronize keys and tensor shapes from all ranks ---
        local_scalar_keys = sorted(metrics_sums.keys())
        local_tensor_info = {k: v.shape for k, v in (per_neg_sums or {}).items()}
        
        all_keys_and_shapes = [None] * world_size
        # logging.info(f"Rank {rank}: Local scalar keys: {local_scalar_keys}, local tensor info: {local_tensor_info}")
        dist.all_gather_object(all_keys_and_shapes, (local_scalar_keys, local_tensor_info))

        # --- 2. Create unified structures on all ranks ---
        unified_scalar_keys = sorted(list(set(k for keys, _ in all_keys_and_shapes for k in keys)))
        
        unified_tensor_info = {}
        for _, tensor_info in all_keys_and_shapes:
            unified_tensor_info.update(tensor_info)
        unified_tensor_keys = sorted(unified_tensor_info.keys())

        # --- 3. Pad local metrics to match unified structures ---
        for k in unified_scalar_keys:
            if k not in metrics_sums:
                metrics_sums[k] = 0.0
        
        if per_neg_sums is None:
            per_neg_sums = {}
        for k in unified_tensor_keys:
            if k not in per_neg_sums:
                shape = unified_tensor_info[k]
                # Ensure tensors are created on the correct device for this rank
                correct_device = device
                if torch.cuda.is_available() and device.type == 'cuda':
                    current_cuda_device = torch.cuda.current_device()
                    correct_device = torch.device(f'cuda:{current_cuda_device}')
                per_neg_sums[k] = torch.zeros(shape, device=correct_device)
        
        # logging.info(f"Rank {rank}: Unified scalar keys: {unified_scalar_keys}, unified tensor keys: {unified_tensor_keys}")
        # --- 4. Aggregate all metrics ---
        aggregated_metrics, total_batches, num_samples_global = ddp_sum_scalar_metrics(
            metrics_sums, unified_scalar_keys, num_batches, num_samples_local, device
        )
        # logging.info(f"Rank {rank}: Aggregated metrics: {aggregated_metrics}")
        aggregated_per_neg = ddp_sum_tensor_metrics(
            per_neg_sums, unified_tensor_keys, device
        )
        # logging.info(f"Rank {rank}: Aggregated per-neg metrics: {aggregated_per_neg}")
        # --- 5. Populate results on rank 0 ---
        if rank == 0:
            metrics_sums = aggregated_metrics
            per_neg_sums = aggregated_per_neg
        else:
            # Other ranks clear their data, no longer needed
            metrics_sums, per_neg_sums = {}, {}
            num_samples_global = 0
            total_batches = 0
    else:
        total_batches = num_batches
        num_samples_global = num_samples_local

    # ---- Normalize & package results (rank 0 or single-process) -----------------
    if not distributed or rank == 0:
        # Avoid division by zero
        nb = max(1, total_batches)
        ns = max(1, num_samples_global)

        # Normalize metrics appropriately
        for k in list(metrics_sums.keys()):
            if k in ['val_loss']:
                # Loss metrics: average over batches
                metrics_sums[k] /= nb
            elif k in ['contrastive_accuracy', 'pos_greater_than_neg']:
                # Count-based metrics: these are sums from get_contrastive_accuracy(get_average=False)
                # Normalize to [0,1] range by dividing by total samples
                metrics_sums[k] = metrics_sums[k] / ns  # Convert sums to normalized accuracy (0-1)
            elif k in ['val_accuracy']:
                # Accuracy from loss function: average over batches
                metrics_sums[k] /= nb
            else:
                # Similarity metrics: average over samples
                metrics_sums[k] /= ns

        if per_neg_sums is None:
            per_neg_sums = {}
        else:
            for k in list(per_neg_sums.keys()):
                # Per-negative metrics: these are sums over all samples, divide by total samples
                per_neg_sums[k] = (per_neg_sums[k] / ns).cpu().numpy()

        # Modality gap placeholders (kept as in your original)

        results = {
            **metrics_sums,
        }
        for k, v in per_neg_sums.items():
            results[k] = v

        if rank == 0:
            logging.info(f"Evaluation aggregated over {world_size} processes and {ns} samples: Results : {results}")
        return results
    else:
        # Non-rank-0 returns minimal stub to keep callers happy
        return {
            'val_loss': 0.0,
            'val_accuracy': 0.0,
        }

def evaluate_ft(model, val_loader, device, loss_fn, **loss_kwargs) -> Dict[str, Any]:
    """Shorthand for fine-tune evaluation."""
    return evaluate(model, val_loader, device, loss_fn, is_finetune=True, **loss_kwargs)