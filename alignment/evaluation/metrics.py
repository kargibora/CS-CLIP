"""Metric computation functions."""

from typing import Any, Callable, Dict

import torch

from utils.evaluate import (
    get_caption_image_similarity,
    get_contrastive_accuracy,
    get_negative_similarity,
    get_negative_similarity_img,
)

from .constants import MetricKeys


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
    Supports dictionary loss outputs and multi-caption mode.
    
    Args:
        img_emb: Image embeddings [B, D]
        pos_txt_emb: Positive text embeddings [B, D] or [B, 1+N, D] for multi-caption mode
        neg_txt_emb: Negative text embeddings [B, D]
        neg_txt_emb_all: All negative text embeddings [B, K, D]
        loss_fn: Loss function
        model: Model (for temperature)
        device: Device for computations
        **loss_kwargs: Additional loss function arguments
        
    Returns:
        Dictionary of computed metrics
    """
    # Ensure all tensors are on the same device
    img_emb = img_emb.to(device)
    pos_txt_emb = pos_txt_emb.to(device)
    neg_txt_emb = neg_txt_emb.to(device)
    neg_txt_emb_all = neg_txt_emb_all.to(device)
    
    metrics = {}
    
    # Check if multi-caption mode (pos_txt_emb is 3D)
    is_multi_caption = pos_txt_emb.dim() == 3
    
    if is_multi_caption:
        # pos_txt_emb shape: [B, 1+N, D]
        # Extract full caption (index 0) for standard metrics
        full_caption_emb = pos_txt_emb[:, 0, :]  # [B, D]
        component_embs = pos_txt_emb[:, 1:, :]  # [B, N, D]
        num_components = component_embs.shape[1]
    else:
        # Standard mode: [B, D]
        full_caption_emb = pos_txt_emb
        component_embs = None
        num_components = 0
    
    # Loss and "accuracy" from loss function (returns a dict)
    # Pass the full pos_txt_emb (could be 2D or 3D) to loss function
    loss_dict = loss_fn(
        img_emb.squeeze(1) if img_emb.dim() == 3 else img_emb,
        pos_text_embeddings=pos_txt_emb,
        neg_text_embeddings=neg_txt_emb,
        temperature=model.temperature,
        device=device,
        **loss_kwargs
    )
    
    # Merge all loss dict keys with 'val_' prefix
    for k, v in loss_dict.items():
        metrics[f"val_{k}"] = v if not torch.is_tensor(v) else v.detach()
    
    # Compute standard metrics using full caption only
    contrastive_acc, n_pos_gt_neg, per_neg_acc = get_contrastive_accuracy(
        img_emb, full_caption_emb, neg_txt_emb_all, get_average=False
    )
    neg_text_sim, per_neg_text_sim = get_negative_similarity(
        full_caption_emb, neg_txt_emb_all, get_average=False
    )
    neg_text_pos_image_sim, per_neg_text_pos_image_sim = get_negative_similarity_img(
        img_emb, neg_txt_emb_all, get_average=False
    )
    pos_sim = get_caption_image_similarity(full_caption_emb, img_emb, get_average=False)

    metrics[MetricKeys.CONTRASTIVE_ACCURACY] = contrastive_acc
    metrics[MetricKeys.POS_GREATER_THAN_NEG] = n_pos_gt_neg
    metrics[MetricKeys.NEG_TEXT_SIMILARITY] = neg_text_sim
    metrics[MetricKeys.NEG_TEXT_POS_IMAGE_SIMILARITY] = neg_text_pos_image_sim
    metrics[MetricKeys.POS_SIMILARITY] = pos_sim
    metrics[MetricKeys.PER_NEG_ACCURACY] = per_neg_acc
    metrics[MetricKeys.PER_NEG_TEXT_SIMILARITY] = per_neg_text_sim
    metrics[MetricKeys.PER_NEG_TEXT_POS_IMAGE_SIMILARITY] = per_neg_text_pos_image_sim
    metrics[MetricKeys.POS_NEG_SIMILARITY_GAP] = pos_sim - neg_text_pos_image_sim
    
    # Multi-caption specific metrics
    if is_multi_caption and num_components > 0:
        # 1. Average similarity of component captions with image
        component_img_sims = []
        for i in range(num_components):
            comp_emb = component_embs[:, i, :]  # [B, D]
            comp_img_sim = get_caption_image_similarity(comp_emb, img_emb, get_average=False)
            component_img_sims.append(comp_img_sim)
        
        avg_component_img_sim = sum(component_img_sims) / num_components
        metrics['component_image_similarity'] = avg_component_img_sim
        
        # 2. Average similarity of component captions with full caption
        component_full_sims = []
        for i in range(num_components):
            comp_emb = component_embs[:, i, :]  # [B, D]
            # Compute cosine similarity between component and full caption
            comp_full_sim = torch.sum(comp_emb * full_caption_emb, dim=-1).mean().item()
            component_full_sims.append(comp_full_sim)
        
        avg_component_full_sim = sum(component_full_sims) / num_components
        metrics['component_full_caption_similarity'] = avg_component_full_sim
        
        # 3. Average similarity of negatives with full caption (already computed above as neg_text_sim)
        # But let's add it explicitly for clarity
        metrics['neg_full_caption_similarity'] = neg_text_sim
        
        # 4. Average similarity of negative caption embeddings with component embeddings
        # Compute similarity between each negative and all components, then average
        component_neg_sims = []
        for i in range(num_components):
            comp_emb = component_embs[:, i, :]  # [B, D]
            # Compute similarity with all negatives for this component
            comp_neg_sim, _ = get_negative_similarity(comp_emb, neg_txt_emb_all, get_average=False)
            component_neg_sims.append(comp_neg_sim)
        
        # Average across all components (sum of sums divided by num_components)
        avg_component_neg_sim = sum(component_neg_sims) / num_components
        metrics['component_neg_similarity'] = avg_component_neg_sim
        
        # 4. Contrastive accuracy: full caption vs component captions
        # For each image, check if full caption has higher similarity than all components
        full_img_sim = torch.sum(full_caption_emb * img_emb, dim=-1)  # [B]
        
        component_beats_full = 0
        for i in range(num_components):
            comp_emb = component_embs[:, i, :]  # [B, D]
            comp_img_sim = torch.sum(comp_emb * img_emb, dim=-1)  # [B]
            component_beats_full += (comp_img_sim > full_img_sim).float().sum().item()
        
        # Accuracy: fraction of times full caption beats all components
        total_comparisons = img_emb.shape[0] * num_components
        full_caption_wins = total_comparisons - component_beats_full
        metrics['full_vs_component_accuracy'] = full_caption_wins / total_comparisons if total_comparisons > 0 else 0.0
        
        # 5. Contrastive accuracy: full caption vs negative captions
        # This is similar to standard contrastive accuracy but explicitly for full caption
        full_vs_neg_correct = 0
        batch_size = full_caption_emb.shape[0]
        K = neg_txt_emb_all.shape[1]
        
        for i in range(K):
            neg_emb = neg_txt_emb_all[:, i, :]  # [B, D]
            neg_img_sim = torch.sum(neg_emb * img_emb, dim=-1)  # [B]
            full_vs_neg_correct += (full_img_sim > neg_img_sim).float().sum().item()
        
        total_neg_comparisons = batch_size * K
        metrics['full_vs_neg_accuracy'] = full_vs_neg_correct / total_neg_comparisons if total_neg_comparisons > 0 else 0.0

    return metrics


def compute_all_metrics_labclip(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    all_neg_scores: torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    **loss_kwargs
) -> Dict[str, Any]:
    """
    Computes metrics for LabCLIP using scores instead of embeddings.
    
    Args:
        pos_scores: Positive pair scores
        neg_scores: Negative pair scores
        all_neg_scores: All negative scores
        loss_fn: Loss function
        device: Device for computations
        **loss_kwargs: Additional loss function arguments
        
    Returns:
        Dictionary of computed metrics
    """
    # Ensure all tensors are on the same device
    pos_scores = pos_scores.to(device)
    neg_scores = neg_scores.to(device)
    all_neg_scores = all_neg_scores.to(device)
    
    metrics = {}
    
    # Loss and accuracy from loss function
    loss_dict = loss_fn(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        device=device,
        **loss_kwargs
    )
    
    # Merge all loss dict keys
    for k, v in loss_dict.items():
        metrics[f"val_{k}"] = v if not torch.is_tensor(v) else v.detach()
    
    # Compute LabCLIP-specific metrics using scores
    
    # 1. Contrastive accuracy: how many times pos_score > neg_score
    contrastive_correct = (pos_scores > neg_scores).float().sum()
    
    # 2. Positive vs all negatives accuracy
    pos_scores_expanded = pos_scores.unsqueeze(1)  # (batch, 1)
    pos_gt_all_negs = (pos_scores_expanded > all_neg_scores).float()  # (batch, K)
    
    # For each sample, check if positive beats ALL negatives
    pos_beats_all_per_sample = pos_gt_all_negs.all(dim=1).float()
    pos_beats_all_count = pos_beats_all_per_sample.sum()
    
    # Per-negative position accuracy
    per_neg_acc = pos_gt_all_negs.sum(dim=0)  # (K,)
    
    # 3. Score statistics
    pos_score_sum = pos_scores.sum()
    neg_score_sum = neg_scores.sum()
    all_neg_score_sum = all_neg_scores.sum()
    score_margin_sum = (pos_scores - neg_scores).sum()
    
    # 4. Per-negative score statistics
    per_neg_score_sums = all_neg_scores.sum(dim=0)  # (K,)
    
    # Store metrics (using sums for proper distributed aggregation)
    metrics[MetricKeys.CONTRASTIVE_ACCURACY] = contrastive_correct
    metrics[MetricKeys.POS_GREATER_THAN_ALL_NEGS] = pos_beats_all_count
    metrics[MetricKeys.POS_SCORE_SUM] = pos_score_sum
    metrics[MetricKeys.NEG_SCORE_SUM] = neg_score_sum
    metrics[MetricKeys.ALL_NEG_SCORE_SUM] = all_neg_score_sum
    metrics[MetricKeys.SCORE_MARGIN_SUM] = score_margin_sum
    metrics[MetricKeys.PER_NEG_ACCURACY] = per_neg_acc
    metrics[MetricKeys.PER_NEG_SCORE_SUMS] = per_neg_score_sums

    return metrics


def normalize_metrics(
    metrics_sums: Dict[str, float],
    per_neg_sums: Dict[str, torch.Tensor],
    num_batches: int,
    num_samples: int
) -> Dict[str, Any]:
    """
    Normalize accumulated metrics appropriately.
    
    Args:
        metrics_sums: Accumulated scalar metrics
        per_neg_sums: Accumulated per-negative metrics
        num_batches: Total number of batches
        num_samples: Total number of samples
        
    Returns:
        Dictionary of normalized metrics
    """
    nb = max(1, num_batches)
    ns = max(1, num_samples)

    results = {}
    
    # Normalize scalar metrics
    for k in list(metrics_sums.keys()):
        if k in [MetricKeys.VAL_LOSS] or k.startswith('val_') and 'loss' in k:
            # Loss metrics: average over batches (includes val_loss, val_component_loss, val_alignment_loss, etc.)
            results[k] = metrics_sums[k] / nb
        elif k in [MetricKeys.CONTRASTIVE_ACCURACY, MetricKeys.POS_GREATER_THAN_NEG, MetricKeys.POS_GREATER_THAN_ALL_NEGS]:
            # Count-based metrics: normalize to [0,1] range
            results[k] = metrics_sums[k] / ns
        elif k in [MetricKeys.VAL_ACCURACY] or (k.startswith('val_') and 'accuracy' in k):
            # Accuracy from loss function: average over batches (includes val_accuracy, val_full_accuracy, etc.)
            results[k] = metrics_sums[k] / nb
        elif k.endswith('_sum'):
            # Convert sums to averages
            avg_key = k.replace('_sum', '_mean')
            results[avg_key] = metrics_sums[k] / ns
        elif k in ['component_image_similarity', 'component_full_caption_similarity', 
                   'neg_full_caption_similarity', 'component_neg_similarity',
                   'full_vs_component_accuracy', 'full_vs_neg_accuracy']:
            # Multi-caption metrics: average over samples
            results[k] = metrics_sums[k] / ns
        else:
            # Similarity metrics: average over samples
            results[k] = metrics_sums[k] / ns
    
    # Normalize per-negative metrics
    if per_neg_sums:
        for k, v in per_neg_sums.items():
            results[k] = (v / ns).cpu().numpy()
    
    return results
