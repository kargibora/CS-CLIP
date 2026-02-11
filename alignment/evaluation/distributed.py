"""Distributed training utilities for metric aggregation."""

from typing import Dict

import torch
import torch.distributed as dist


def ddp_sum_scalar_metrics(
    metrics_sums: Dict[str, float],
    unified_keys: list,
    num_batches: int,
    num_samples_local: int,
    device: torch.device
):
    """
    Reduce scalar metrics across ranks using all_reduce (sum).
    
    Args:
        metrics_sums: Local metric sums
        unified_keys: Synchronized metric keys across all ranks
        num_batches: Number of batches processed locally
        num_samples_local: Number of samples processed locally
        device: Device for tensor operations
        
    Returns:
        Tuple of (aggregated_metrics, total_batches, num_samples_global)
    """
    if not unified_keys:
        buffer = torch.tensor([float(num_batches), float(num_samples_local)], dtype=torch.float32, device=device)
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        total_batches = int(buffer[0].item())
        num_samples_global = int(buffer[1].item())
        return {}, total_batches, num_samples_global

    # Ensure all keys exist in the local dict
    for k in unified_keys:
        metrics_sums.setdefault(k, 0.0)

    # Order: num_batches, num_samples, then all metrics_sums alphabetically
    values = [float(num_batches), float(num_samples_local)] + [metrics_sums[k] for k in unified_keys]
    buffer = torch.tensor(values, dtype=torch.float32, device=device)
    
    # Ensure buffer is on the correct device for this rank
    if torch.cuda.is_available() and device.type == 'cuda':
        current_device = torch.cuda.current_device()
        if buffer.device.index != current_device:
            buffer = buffer.to(f'cuda:{current_device}')
    
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    
    # Unpack summed values
    total_batches = int(buffer[0].item())
    num_samples_global = int(buffer[1].item())
    
    # Reconstruct the metrics dict
    aggregated_metrics = {k: buffer[i+2].item() for i, k in enumerate(unified_keys)}
    
    return aggregated_metrics, total_batches, num_samples_global


def ddp_sum_tensor_metrics(
    per_neg_sums: Dict[str, torch.Tensor],
    unified_keys: list,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Reduce per-neg tensors using all_reduce.
    
    Args:
        per_neg_sums: Local per-negative metric sums
        unified_keys: Synchronized metric keys across all ranks
        device: Device for tensor operations
        
    Returns:
        Aggregated per-negative metrics
    """
    if not unified_keys:
        return {}

    # All-reduce each tensor
    for k in unified_keys:
        tensor = per_neg_sums[k].to(device)
        
        # Ensure tensor is on the correct device for this rank
        if torch.cuda.is_available() and device.type == 'cuda':
            current_device = torch.cuda.current_device()
            if tensor.device.index != current_device:
                tensor = tensor.to(f'cuda:{current_device}')
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        per_neg_sums[k] = tensor
    
    return per_neg_sums


def aggregate_distributed_metrics(
    metrics_sums: Dict[str, float],
    per_neg_sums: Dict[str, torch.Tensor],
    num_batches: int,
    num_samples_local: int,
    device: torch.device,
    world_size: int,
    rank: int
):
    """
    Aggregate metrics across all distributed processes.
    
    Args:
        metrics_sums: Local scalar metric sums
        per_neg_sums: Local per-negative metric sums
        num_batches: Number of batches processed locally
        num_samples_local: Number of samples processed locally
        device: Device for tensor operations
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        Tuple of (aggregated_metrics, aggregated_per_neg, total_batches, num_samples_global)
    """
    dist.barrier()

    # Synchronize keys and tensor shapes from all ranks
    local_scalar_keys = sorted(metrics_sums.keys())
    local_tensor_info = {k: v.shape for k, v in (per_neg_sums or {}).items()}
    
    all_keys_and_shapes = [None] * world_size
    dist.all_gather_object(all_keys_and_shapes, (local_scalar_keys, local_tensor_info))

    # Create unified structures on all ranks
    unified_scalar_keys = sorted(list(set(k for keys, _ in all_keys_and_shapes for k in keys)))
    
    unified_tensor_info = {}
    for _, tensor_info in all_keys_and_shapes:
        unified_tensor_info.update(tensor_info)
    unified_tensor_keys = sorted(unified_tensor_info.keys())

    # Pad local metrics to match unified structures
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

    # Aggregate all metrics
    aggregated_metrics, total_batches, num_samples_global = ddp_sum_scalar_metrics(
        metrics_sums, unified_scalar_keys, num_batches, num_samples_local, device
    )
    aggregated_per_neg = ddp_sum_tensor_metrics(
        per_neg_sums, unified_tensor_keys, device
    )
    
    # Populate results on rank 0
    if rank == 0:
        return aggregated_metrics, aggregated_per_neg, total_batches, num_samples_global
    else:
        # Other ranks clear their data
        return {}, {}, 0, 0
