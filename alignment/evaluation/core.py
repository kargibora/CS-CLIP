"""Core evaluation functions."""

import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from .constants import MetricKeys, ModelType
from .distributed import aggregate_distributed_metrics
from .metrics import (
    compute_all_metrics,
    compute_all_metrics_labclip,
    normalize_metrics,
)
from .outputs import (
    gather_validation_outputs,
    gather_validation_outputs_multilayer,
    is_tca_model,
)
from .utils import (
    ensure_correct_device,
    is_labclip_model,
    is_ft_mode,
)
from .visualization import (
    SimilarityVisualizer,
    collect_visualization_batch,
)


def evaluate(
    model: Any,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Callable,
    is_finetune: bool = False,
    **loss_kwargs
) -> Dict[str, Any]:
    """
    Main evaluation function for standard CLIP models.
    
    Args:
        model: Model to evaluate
        val_loader: Validation dataloader
        device: Device for computations
        loss_fn: Loss function
        is_finetune: Whether this is fine-tuning mode
        **loss_kwargs: Additional loss function arguments
        
    Returns:
        Dictionary of evaluation metrics
    """

    # Original evaluation pipeline for standard CLIP models
    model_class = model.module.__class__.__name__ if hasattr(model, 'module') else model.__class__.__name__
    logging.info(f"Using standard evaluation pipeline for {model_class}")
    
    model.eval()
    distributed = dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0

    # Device management for distributed training
    if distributed and torch.cuda.is_available():
        device = ensure_correct_device(device, rank)
        torch.cuda.set_device(torch.cuda.current_device())

    metrics_sums: Dict[str, float] = {}
    per_neg_sums: Dict[str, torch.Tensor] = None
    num_batches = 0
    num_samples_local = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", disable=(rank != 0))):
            try:
                # Route to appropriate output gathering function
                if model.__class__.__name__ == ModelType.FLEXIBLE_CLIP_MULTILAYER:
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
                raise e
                continue

    # Distributed aggregation
    if distributed:
        metrics_sums, per_neg_sums, num_batches, num_samples_global = aggregate_distributed_metrics(
            metrics_sums, per_neg_sums, num_batches, num_samples_local, device, world_size, rank
        )
    else:
        num_samples_global = num_samples_local

    # Normalize & package results (rank 0 or single-process)
    if not distributed or rank == 0:
        results = normalize_metrics(metrics_sums, per_neg_sums, num_batches, num_samples_global)
        
        if rank == 0:
            logging.info(f"Evaluation aggregated over {world_size} processes and {num_samples_global} samples: Results: {results}")
        return results
    else:
        # Non-rank-0 returns minimal stub
        return {
            MetricKeys.VAL_LOSS: 0.0,
            MetricKeys.VAL_ACCURACY: 0.0,
        }



def evaluate_ft(
    model: Any,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Callable,
    **loss_kwargs
) -> Dict[str, Any]:
    """
    Shorthand for fine-tune evaluation.
    
    Args:
        model: Model to evaluate
        val_loader: Validation dataloader
        device: Device for computations
        loss_fn: Loss function
        **loss_kwargs: Additional loss function arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    return evaluate(model, val_loader, device, loss_fn, is_finetune=True, **loss_kwargs)
