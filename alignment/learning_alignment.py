# 0.779 --> 0.90
# 0.87 --> 0.936

import clip
import argparse
import logging
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from alignment.evaluation import evaluate, evaluate_ft
# Remove the old dataset evaluation import since we'll import it locally when needed
from utils.evaluate import flatten_per_neg_metric
from utils.dist import (
    is_main_process,
    MultiGPUWrapper
    )


import torch
import torch.distributed as dist
from collections import defaultdict
import logging


import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
import logging
import os
import clip


# Training function for model with negative examples
class AllGatherVariableBatch(torch.autograd.Function):
    """
    Autograd-safe all_gather for variable local batch sizes using:
    pad -> all_gather (equal shapes) -> slice valid rows -> concat.
    Forward returns [sum_i bs_i, *trailing].
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        assert dist.is_initialized(), "torch.distributed must be initialized"
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 1) Gather per-rank batch sizes (no grad needed)
        bs_local = x.shape[0]
        bs_t = torch.tensor([bs_local], device=x.device, dtype=torch.long)
        all_bs_list = [torch.zeros_like(bs_t) for _ in range(world_size)]
        dist.all_gather(all_bs_list, bs_t)
        batch_sizes = [int(t.item()) for t in all_bs_list]

        total_valid = sum(batch_sizes)
        trailing = x.shape[1:]

        if total_valid == 0:
            # everyone empty
            ctx.local_bs = 0
            ctx.offset = 0
            ctx.trailing = trailing
            ctx.batch_sizes = batch_sizes
            return x.new_zeros((0, *trailing))

        max_bs = max(batch_sizes)

        # 2) Pad locally to max_bs
        if bs_local < max_bs:
            pad = x.new_zeros((max_bs - bs_local, *trailing))
            x_pad = torch.cat([x, pad], dim=0)
        else:
            x_pad = x

        # 3) all_gather with equal-shaped tensors
        chunks = [x_pad.new_empty((max_bs, *trailing)) for _ in range(world_size)]
        dist.all_gather(chunks, x_pad)

        # 4) strip padding and concat valid rows (rank order)
        valid = [chunks[i][:batch_sizes[i]] for i in range(world_size) if batch_sizes[i] > 0]
        y = torch.cat(valid, dim=0)

        # Save slice info for backward
        offsets = [0]
        for bs in batch_sizes[:-1]:
            offsets.append(offsets[-1] + bs)

        ctx.local_bs = batch_sizes[rank]
        ctx.offset = offsets[rank]
        ctx.trailing = trailing
        ctx.batch_sizes = batch_sizes
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        if ctx.local_bs == 0:
            return grad_y.new_zeros((0, *ctx.trailing))
        start = ctx.offset
        end = start + ctx.local_bs
        return grad_y[start:end].contiguous()


# ---------- Optimized autograd Function (all_gather_into_tensor) ----------

class AllGatherVariableBatchOptimized(torch.autograd.Function):
    """
    Same semantics as above, but uses dist.all_gather_into_tensor for lower overhead.
    Requires PyTorch with dist.all_gather_into_tensor (>= 1.11; better >= 2.1).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        assert dist.is_initialized(), "torch.distributed must be initialized"
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 1) Gather per-rank batch sizes
        bs_local = x.shape[0]
        bs_t = torch.tensor([bs_local], device=x.device, dtype=torch.long)
        bs_all = torch.empty(world_size, dtype=torch.long, device=x.device)
        dist.all_gather_into_tensor(bs_all, bs_t)  # -> [world_size]
        batch_sizes = bs_all.tolist()

        total_valid = int(bs_all.sum().item())
        trailing = x.shape[1:]

        if total_valid == 0:
            ctx.local_bs = 0
            ctx.offset = 0
            ctx.trailing = trailing
            ctx.batch_sizes = batch_sizes
            return x.new_zeros((0, *trailing))

        max_bs = int(bs_all.max().item())

        # 2) Pad locally to max_bs
        if bs_local < max_bs:
            pad = x.new_zeros((max_bs - bs_local, *trailing))
            x_pad = torch.cat([x, pad], dim=0)
        else:
            x_pad = x

        # 3) Gather into a single tensor of shape [world_size, max_bs, *trailing]
        gathered = torch.empty((world_size, max_bs, *trailing), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered, x_pad)

        # 4) strip padding and concat valid rows (rank order)
        valid = []
        for i, bs in enumerate(batch_sizes):
            if bs > 0:
                valid.append(gathered[i, :bs])  # shape [bs, *trailing]
        y = torch.cat(valid, dim=0)

        # Save slice info for backward
        offsets = [0]
        for bs in batch_sizes[:-1]:
            offsets.append(offsets[-1] + bs)

        ctx.local_bs = batch_sizes[rank]
        ctx.offset = offsets[rank]
        ctx.trailing = trailing
        ctx.batch_sizes = batch_sizes
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        if ctx.local_bs == 0:
            return grad_y.new_zeros((0, *ctx.trailing))
        start = ctx.offset
        end = start + ctx.local_bs
        return grad_y[start:end].contiguous()


# ---------- Public helper ----------

def all_gather_variable_batch_with_grad(x: torch.Tensor, use_optimized: bool = True):
    """
    Returns:
        y: Tensor of shape [sum_i bs_i, *trailing], identical on all ranks.
        batch_sizes: Python list of per-rank sizes [bs_0, ..., bs_{p-1}].
    """
    if not dist.is_initialized():
        # Single process fallback
        return x, [int(x.shape[0])]

    # Use optimized path if available
    if use_optimized and hasattr(dist, "all_gather_into_tensor"):
        y = AllGatherVariableBatchOptimized.apply(x)
    else:
        y = AllGatherVariableBatch.apply(x)

    # Also return the batch sizes so caller can know N and per-rank offsets
    # (gather once more; cheap and avoids relying on ctx outside the Function)
    world_size = dist.get_world_size()
    bs_t = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
    if hasattr(dist, "all_gather_into_tensor"):
        bs_all = torch.empty(world_size, dtype=torch.long, device=x.device)
        dist.all_gather_into_tensor(bs_all, bs_t)
        batch_sizes = bs_all.tolist()
    else:
        lst = [torch.zeros_like(bs_t) for _ in range(world_size)]
        dist.all_gather(lst, bs_t)
        batch_sizes = [int(t.item()) for t in lst]

    return y, batch_sizes

def synchronize_batch_processing(device, batch_idx, continue_training):
    """
    Synchronize batch processing across GPUs to handle uneven batch counts.
    Returns True if all GPUs should continue, False if any GPU is done.
    """
    world_size = dist.get_world_size()
    
    # Each GPU broadcasts whether it wants to continue
    continue_flags = [torch.zeros(1, device=device, dtype=torch.uint8) for _ in range(world_size)]
    local_flag = torch.tensor([1 if continue_training else 0], device=device, dtype=torch.uint8)
    dist.all_gather(continue_flags, local_flag)
    
    # All GPUs must have batches to continue
    all_continue = all(flag.item() > 0 for flag in continue_flags)
    
    return all_continue


def compute_contrastive_loss_with_merged_batches(
    image_embeddings, 
    text_embeddings, 
    neg_text_embeddings, 
    temperature, 
    device,
    loss_fn,
    skip_batch: bool = False,
    loss_kwargs: dict = None,
    components_per_caption = None,
    num_components_available = None,
    caption_valid_mask = None,
    paraphrase_embeddings = None,
    has_paraphrase = None,
):
    """
    Compute contrastive loss after merging batches from all distributed workers.
    Assumes `loss_fn` expects *global* (gathered) embeddings and uses global N for labels.
    
    Args:
        components_per_caption: Optional[torch.Tensor] [B, N] - components in each component caption
        num_components_available: Optional[torch.Tensor] [B] - total components available for each sample
        caption_valid_mask: Optional[torch.Tensor] [B, 1+N] - validity mask for each caption
        paraphrase_embeddings: Optional[torch.Tensor] [B, D] - paraphrase text embeddings
        has_paraphrase: Optional[torch.Tensor] [B] - boolean mask for samples with paraphrases
    """
    if loss_kwargs is None:
        loss_kwargs = {}

    local_batch_size = 0 if skip_batch else int(image_embeddings.shape[0])

    # Early exit for empty/skip
    if skip_batch or local_batch_size == 0:
        dummy_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        return {
            'loss': dummy_loss,
            'accuracy': 0.0,
            'global_batch_size': 0,
            'local_batch_size': 0,
            'skip_batch': True
        }, 0

    if dist.is_initialized() and dist.get_world_size() > 1:
        # Detect multi-caption mode (3D tensors: [B, 1+N, D])
        is_multi_caption = text_embeddings.dim() == 3
        N_cap = text_embeddings.shape[1] if is_multi_caption else 1
        
        if is_multi_caption:
            # Flatten 3D [B, 1+N, D] -> 2D [B*(1+N), D] for all_gather
            B_local, _, D = text_embeddings.shape
            txt_flat = text_embeddings.view(B_local * N_cap, D).contiguous()
            neg_flat = neg_text_embeddings.view(B_local * N_cap, D).contiguous()
            
            # Gather all three streams
            img_all, img_sizes = all_gather_variable_batch_with_grad(image_embeddings)  # [B_total, D]
            txt_gathered, txt_sizes = all_gather_variable_batch_with_grad(txt_flat)  # [B_total*N_cap, D]
            neg_gathered, neg_sizes = all_gather_variable_batch_with_grad(neg_flat)  # [B_total*N_cap, D]
            
            # Unflatten back to 3D: [B_total*(1+N), D] -> [B_total, 1+N, D]
            total_B = sum(img_sizes)
            txt_all = txt_gathered.view(total_B, N_cap, D)
            neg_all = neg_gathered.view(total_B, N_cap, D)
        else:
            # Standard 2D mode [B, D]
            img_all, img_sizes = all_gather_variable_batch_with_grad(image_embeddings)
            txt_all, txt_sizes = all_gather_variable_batch_with_grad(text_embeddings)
            neg_all, neg_sizes = all_gather_variable_batch_with_grad(neg_text_embeddings)
        
        # Gather coverage information if present
        components_all = None
        num_available_all = None
        valid_mask_all = None
        paraphrase_all = None
        has_paraphrase_all = None
        if components_per_caption is not None:
            # components_per_caption is [B, N], need to gather across GPUs
            # Flatten to [B*N] for gathering, then reshape back
            B, N = components_per_caption.shape
            components_flat = components_per_caption.reshape(-1).float()
            components_gathered, _ = all_gather_variable_batch_with_grad(components_flat)
            # Reshape back to [B_total, N]
            total_B = sum(img_sizes)
            components_all = components_gathered.reshape(total_B, N).long()
        if num_components_available is not None:
            num_available_all, _ = all_gather_variable_batch_with_grad(num_components_available.float())
            num_available_all = num_available_all.long()  # Convert back to long
        if caption_valid_mask is not None:
            # caption_valid_mask is [B, 1+N], need to gather across GPUs
            B, M = caption_valid_mask.shape
            valid_flat = caption_valid_mask.reshape(-1).float()  # Convert bool to float for gather
            valid_gathered, _ = all_gather_variable_batch_with_grad(valid_flat)
            total_B = sum(img_sizes)
            valid_mask_all = valid_gathered.reshape(total_B, M).bool()  # Convert back to bool
        
        # Gather paraphrase embeddings if present
        if paraphrase_embeddings is not None:
            paraphrase_all, _ = all_gather_variable_batch_with_grad(paraphrase_embeddings)
        if has_paraphrase is not None:
            has_paraphrase_float = has_paraphrase.float()  # Convert bool to float for gather
            has_paraphrase_gathered, _ = all_gather_variable_batch_with_grad(has_paraphrase_float)
            has_paraphrase_all = has_paraphrase_gathered.bool()  # Convert back to bool

        N_img = sum(img_sizes)
        N_txt = sum(txt_sizes)
        N_neg = sum(neg_sizes)

        # Strict sanity check - differs between multi-caption and standard mode
        if is_multi_caption:
            # In multi-caption mode: txt/neg were flattened, so N_txt == N_img * N_cap
            # After unflattening: txt_all.size(0) == N_img and txt_all.shape[1] == N_cap
            expected_txt_size = N_img * N_cap
            if not (N_txt == N_neg == expected_txt_size):
                raise RuntimeError(
                    f"Global batch mismatch (multi-caption): N_img={N_img}, N_txt={N_txt}, N_neg={N_neg}, "
                    f"expected txt/neg size={expected_txt_size} (N_img * N_cap)"
                )
            if not (img_all.size(0) == txt_all.size(0) == neg_all.size(0) == N_img):
                raise RuntimeError(
                    f"Global batch mismatch after unflatten: img={img_all.size(0)}, txt={txt_all.size(0)}, neg={neg_all.size(0)}, expected={N_img}"
                )
        else:
            # Standard mode: all sizes should match
            if not (N_img == N_txt == N_neg == img_all.size(0) == txt_all.size(0) == neg_all.size(0)):
                raise RuntimeError(
                    f"Global batch mismatch: N_img={N_img}, N_txt={N_txt}, N_neg={N_neg}, "
                    f"gathered sizes: img={img_all.size(0)}, txt={txt_all.size(0)}, neg={neg_all.size(0)}"
                )

        if N_img > 0:
            # Add coverage to loss_kwargs if present
            if components_all is not None:
                loss_kwargs = {**loss_kwargs, 'components_per_caption': components_all}
            if num_available_all is not None:
                loss_kwargs = {**loss_kwargs, 'num_components_available': num_available_all}
            if valid_mask_all is not None:
                loss_kwargs = {**loss_kwargs, 'caption_valid_mask': valid_mask_all}
            # Add paraphrase embeddings if present
            if paraphrase_all is not None:
                loss_kwargs = {**loss_kwargs, 'paraphrase_embeddings': paraphrase_all}
            if has_paraphrase_all is not None:
                loss_kwargs = {**loss_kwargs, 'has_paraphrase': has_paraphrase_all}
            
            loss_dict = loss_fn(
                image_embeddings=img_all,
                pos_text_embeddings=txt_all,
                neg_text_embeddings=neg_all,
                temperature=temperature,
                device=device,
                **loss_kwargs
            )
        else:
            loss_dict = {
                'loss': torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True),
                'accuracy': 0.0
            }

        loss_dict['global_batch_size'] = N_img
        loss_dict['local_batch_size'] = local_batch_size
        return loss_dict, local_batch_size

    # Single-process
    # Add coverage to loss_kwargs if present
    if components_per_caption is not None:
        loss_kwargs = {**loss_kwargs, 'components_per_caption': components_per_caption}
    if num_components_available is not None:
        loss_kwargs = {**loss_kwargs, 'num_components_available': num_components_available}
    if caption_valid_mask is not None:
        loss_kwargs = {**loss_kwargs, 'caption_valid_mask': caption_valid_mask}
    # Add paraphrase embeddings if present
    if paraphrase_embeddings is not None:
        loss_kwargs = {**loss_kwargs, 'paraphrase_embeddings': paraphrase_embeddings}
    if has_paraphrase is not None:
        loss_kwargs = {**loss_kwargs, 'has_paraphrase': has_paraphrase}
    
    loss_dict = loss_fn(
        image_embeddings=image_embeddings,
        pos_text_embeddings=text_embeddings,
        neg_text_embeddings=neg_text_embeddings,
        temperature=temperature,
        device=device,
        **loss_kwargs
    )
    loss_dict['global_batch_size'] = local_batch_size
    loss_dict['local_batch_size'] = local_batch_size
    return loss_dict, local_batch_size


def train_model_multigpu_merged_batch(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    evaluate_fn=None,
    scheduler=None,
    metric_key='contrastive_accuracy',
    log_fn=None,
    batch_unpack_fn=None,
    scheduler_on='epoch',
    cfg=None,
    gpu_wrapper=None,
    train_sampler=None,
    clip_model=None,
    preprocess=None,
    loss_kwargs=None,
    sample_logger=None,
    # Config sections - training function extracts parameters from these
    train_cfg=None,
    eval_cfg=None,
    align_cfg=None,
    loss_cfg=None,
    logger_cfg=None,
    **extra_log_params
):
    """
    Multi-GPU training loop with batch merging for contrastive learning.
    Properly handles uneven batch counts across GPUs.
    
    All training parameters are now extracted from config sections (train_cfg, eval_cfg, etc.)
    This allows adding new parameters to Hydra configs without modifying this function signature.
    
    Args:
        train_cfg: Training configuration (epochs, batch_size, etc.)
        eval_cfg: Evaluation configuration (initial_evaluate, enable_dataset_eval, etc.)
        align_cfg: Alignment configuration (alignment_type, etc.)
        loss_cfg: Loss configuration (auxiliary loss weights, etc.)
        logger_cfg: Logger configuration (log_every_k_steps, enable_sample_logging, etc.)
        loss_kwargs: Dict of auxiliary loss parameters (backward compatibility)
    """
    
    # Extract parameters from config sections
    # Training config
    num_epochs = getattr(train_cfg, 'epochs', 10) if train_cfg else 10
    evaluate_every_n = getattr(train_cfg, 'eval_n', 1) if train_cfg else 1
    use_amp = getattr(train_cfg, 'use_amp', False) if train_cfg else False
    save_path = getattr(train_cfg, 'save_path', None) if train_cfg else None
    
    # Evaluation config  
    initial_evaluate = getattr(eval_cfg, 'initial_evaluate', True) if eval_cfg else True
    enable_dataset_eval = getattr(eval_cfg, 'enable_dataset_eval', True) if eval_cfg else True
    dataset_eval_datasets = getattr(eval_cfg, 'dataset_eval_datasets', None) if eval_cfg else None
    dataset_eval_csv_path = getattr(eval_cfg, 'dataset_eval_csv_path', "dataset_evaluation_results.csv") if eval_cfg else "dataset_evaluation_results.csv"
    
    # Alignment config
    alignment_type = getattr(align_cfg, 'alignment_type', "HNB") if align_cfg else "HNB"
    
    # Logger config
    log_every_k_steps = getattr(logger_cfg, 'log_every_k_steps', None) if logger_cfg else None
    eval_every_k_steps = getattr(logger_cfg, 'eval_every_k_steps', None) if logger_cfg else None
    save_every_k_steps = getattr(logger_cfg, 'save_every_k_steps', None) if logger_cfg else None
    sample_every_k_steps = getattr(logger_cfg, 'sample_every_k_steps', None) if logger_cfg else None
    
    # Loss config - extract loss_kwargs from loss_cfg
    if loss_kwargs is None:
        if loss_cfg is not None:
            # Convert OmegaConf to dict for loss_kwargs
            from omegaconf import OmegaConf
            loss_kwargs = OmegaConf.to_container(loss_cfg, resolve=True)
            if not isinstance(loss_kwargs, dict):
                loss_kwargs = {}
        else:
            loss_kwargs = {}
    
    distributed = getattr(cfg.dist, 'distributed', False) if cfg else False
    should_log = is_main_process()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    
    
    if should_log:
        mode_name = 'Standard CLIP'
        logging.info(f"Training mode: {mode_name}")
    
    # Initialize AMP scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Setup dataset evaluation if enabled
    dataset_evaluator = None
    if enable_dataset_eval and dataset_eval_datasets:
        try:
            # Use distributed evaluation in multi-GPU settings
            if distributed and world_size > 1:
                from alignment.distributed_dataset_evaluation import setup_distributed_dataset_evaluation
                dataset_evaluator = setup_distributed_dataset_evaluation(
                    datasets=dataset_eval_datasets,
                    csv_path=dataset_eval_csv_path
                )
                if should_log:
                    print(f"Distributed dataset evaluation enabled for: {dataset_eval_datasets} across {world_size} GPUs")
            else:
                # Single GPU: use standard evaluator
                from alignment.simple_dataset_evaluation import setup_dataset_evaluation
                dataset_evaluator = setup_dataset_evaluation(
                    datasets=dataset_eval_datasets,
                    csv_path=dataset_eval_csv_path
                )
                if should_log:
                    print(f"Dataset evaluation enabled for: {dataset_eval_datasets}")
        except Exception as e:
            if should_log:
                print(f"Failed to setup dataset evaluation: {e}")
            dataset_evaluator = None

    # === Initial evaluation ===
    if initial_evaluate and evaluate_fn is not None:
        base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
        logs = evaluate_fn(base_model, val_loader, device, loss_fn, **loss_kwargs)

        benchmark_results = {}
        
        if dataset_evaluator is not None:
            try:
                if should_log:
                    logging.info("🔍 Starting initial dataset evaluation (before training)")
                benchmark_results = dataset_evaluator.evaluate_all(
                    model=base_model,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    alignment_type=alignment_type,
                    step=0,
                    epoch=0,
                    wandb_log=False,
                    is_initial_eval=True
                )
                if should_log:
                    logging.info(f"✅ Initial dataset evaluation completed: {len(benchmark_results)} results")
                    
                    # Log the actual initial benchmark results for debugging
                    if benchmark_results:
                        logging.info("📊 Initial benchmark results:")
                        for key, value in benchmark_results.items():
                            if isinstance(value, (int, float)):
                                logging.info(f"   {key}: {value:.4f}")
                            else:
                                logging.info(f"   {key}: {value}")
                    else:
                        logging.warning("⚠️ Initial benchmark evaluation returned empty results")
                    
            except Exception as e:
                if should_log:
                    error_msg = f"Initial dataset evaluation failed: {e}"
                    logging.error(f"❌ {error_msg}")
                    import traceback
                    logging.error(f"📍 Initial evaluation traceback:\n{traceback.format_exc()}")
                benchmark_results = {}

        if should_log and log_fn:
            logs.update(benchmark_results)
            # Initial evaluation uses step 0
            try:
                # Try to pass step parameter for wandb synchronization
                log_fn(logs, step=0)
            except TypeError:
                # Fallback for log functions that don't accept step parameter
                log_fn(logs)

    best_metric_value = -float('inf')
    best_model_state = None
    best_epoch = -1
    best_step = -1
    
    # Global step counter across all epochs
    global_step = 0
    
    # Determine step-based logging configuration
    use_step_logging = log_every_k_steps is not None
    use_step_evaluation = eval_every_k_steps is not None
    use_step_saving = save_every_k_steps is not None
    use_step_sample_logging = sample_every_k_steps is not None
    
    if should_log and (use_step_logging or use_step_evaluation or use_step_saving or use_step_sample_logging):
        logging.info("📊 Step-based training enabled:")
        if use_step_logging:
            logging.info(f"   - Logging every {log_every_k_steps} steps")
        if use_step_evaluation:
            logging.info(f"   - Evaluation every {eval_every_k_steps} steps") 
        if use_step_saving:
            logging.info(f"   - Checkpoints every {save_every_k_steps} steps")
        if use_step_sample_logging:
            logging.info(f"   - Sample visualizations every {sample_every_k_steps} steps")

    epoch_iter = tqdm(
        range(num_epochs), 
        desc=f"🔄 Merged Training ({alignment_type})", 
        unit="epoch",
        ncols=100,
        colour='cyan'
    ) if should_log else range(num_epochs)

    for epoch in epoch_iter:
        # Sync sampler shuffling
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        model.train()
        loss_sums = defaultdict(float)
        loss_counts = defaultdict(int)
        train_metrics = []
        
        batch_count = 0
        skipped_batches = 0
        processed_batches = 0

        # Create an iterator that we can control
        train_iter = iter(train_loader)
        has_data = True
        
        # Progress bar for main process
        if should_log:
            total_batches = len(train_loader) if hasattr(train_loader, '__len__') else None
            pbar = tqdm(
                total=total_batches,
                desc=f"⚡ Epoch {epoch+1:3d}/{num_epochs} | Batches", 
                leave=False,
                ncols=150,
                colour='yellow'
            )

        while True:
            # Check if this GPU still has data
            try:
                batch = next(train_iter)
                has_batch = True
            except StopIteration:
                has_batch = False
                # Create dummy batch for GPUs that are done
                batch = None
            
            # Synchronize: check if all GPUs still have batches
            if distributed:
                all_continue = synchronize_batch_processing(device, batch_count, has_batch)
                if not all_continue:
                    # All GPUs are done, so everyone stops
                    logging.info(f"Rank {rank}: Stopping at batch {batch_count} as all GPUs are done.")
                    break
            elif not has_batch:
                # Single GPU case: stop when done
                break
            
            batch_count += 1
            
            # Initialize loss_dict with default values (will be updated if batch is processed)
            loss_dict = {'skip_batch': True, 'loss': torch.tensor(0.0, device=device)}
            
            # Process batch only if this GPU has data
            if has_batch and batch is not None:
                optimizer.zero_grad()
                
                if batch_unpack_fn is None:
                    raise ValueError("batch_unpack_fn must be provided")

                try:
                    # Wrap forward pass with autocast for mixed precision
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        # Unpack batch - different logic for LabCLIP vs TCA vs standard CLIP
                        unpack = batch_unpack_fn(batch, model, device)
                        
                        # Standard CLIP with embeddings
                        pos_text_embeddings = unpack['text_embeddings']
                        neg_text_embeddings = unpack.get('neg_text_embeddings')
                        image_embeddings = unpack['image_embeddings']
                        temperature = unpack['temperature']
                        unpack_device = unpack['device']
                        components_per_caption = unpack.get('components_per_caption')
                        num_components_available = unpack.get('num_components_available')
                        caption_valid_mask = unpack.get('caption_valid_mask')  # [B, 1+N] NEW!
                        # Paraphrase embeddings for sentence alignment loss
                        paraphrase_embeddings = unpack.get('paraphrase_embeddings')  # [B, D] or None
                        has_paraphrase = unpack.get('has_paraphrase')  # [B] bool or None
                        
                        # Validate batch
                        if image_embeddings.shape[0] == 0:
                            logging.warning(f"Rank {rank}: Batch {batch_count} - CLIP image_embeddings has 0 samples")
                            # Create dummy tensors for all_gather
                            image_embeddings = torch.zeros(1, image_embeddings.shape[1], device=device, dtype=image_embeddings.dtype)
                            pos_text_embeddings = torch.zeros(1, pos_text_embeddings.shape[1], device=device, dtype=pos_text_embeddings.dtype)
                            if neg_text_embeddings is not None:
                                neg_text_embeddings = torch.zeros(1, neg_text_embeddings.shape[1], device=device, dtype=neg_text_embeddings.dtype)
                            skip_this_batch = True
                        else:
                            skip_this_batch = False
                        
                        if not skip_this_batch:
                            # Validate embeddings for NaN/Inf
                            if torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any():
                                logging.error(f"Rank {rank}: Batch {batch_count} - NaN/Inf detected in image_embeddings!")
                                skip_this_batch = True
                            elif torch.isnan(pos_text_embeddings).any() or torch.isinf(pos_text_embeddings).any():
                                logging.error(f"Rank {rank}: Batch {batch_count} - NaN/Inf detected in text_embeddings!")
                                skip_this_batch = True
                            elif neg_text_embeddings is not None and neg_text_embeddings.shape[0] > 0:
                                if torch.isnan(neg_text_embeddings).any() or torch.isinf(neg_text_embeddings).any():
                                    logging.error(f"Rank {rank}: Batch {batch_count} - NaN/Inf detected in neg_text_embeddings!")
                                    skip_this_batch = True
                        
                        if neg_text_embeddings is None:
                            neg_text_embeddings = torch.zeros(0, pos_text_embeddings.shape[1], 
                                                                dtype=pos_text_embeddings.dtype, 
                                                                device=device)
                        
                        # Compute standard CLIP loss with embeddings
                        loss_dict, local_batch_size = compute_contrastive_loss_with_merged_batches(
                            image_embeddings=image_embeddings,
                            text_embeddings=pos_text_embeddings,
                            neg_text_embeddings=neg_text_embeddings,
                            temperature=temperature,
                            device=unpack_device,
                            loss_fn=loss_fn,
                            skip_batch=skip_this_batch,
                            loss_kwargs=loss_kwargs,
                            components_per_caption=components_per_caption,
                            num_components_available=num_components_available,
                            caption_valid_mask=caption_valid_mask,
                            paraphrase_embeddings=paraphrase_embeddings,
                            has_paraphrase=has_paraphrase,
                        )
                        
                        # Add temperature and learning rate to loss_dict for progress bar display
                        if not loss_dict.get('skip_batch', False):
                            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                                loss_dict['learning_rate'] = optimizer.param_groups[0].get('lr', 0)
                    
                except Exception as e:
                    logging.error(f"Rank {rank}: Error processing batch {batch_count}: {e}")
                    logging.error(f"Rank {rank}: Batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        logging.error(f"Rank {rank}: Batch keys: {batch.keys()}")
                        for key, value in batch.items():
                            if value is not None:
                                if torch.is_tensor(value):
                                    logging.error(f"Rank {rank}: {key} shape: {value.shape}, dtype: {value.dtype}")
                                else:
                                    logging.error(f"Rank {rank}: {key} type: {type(value)}")
                            else:
                                logging.error(f"Rank {rank}: {key} is None!")
                    else:
                        logging.error(f"Rank {rank}: Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
                    
                    import traceback
                    logging.error(f"Rank {rank}: Traceback:\n{traceback.format_exc()}")
                    
                    # Create dummy loss for error case
                    loss_dict = {
                        'loss': torch.tensor(0.0, device=device, requires_grad=True),
                        'accuracy': 0.0,
                        'skip_batch': True,
                        'skip_reason': f'Exception: {str(e)[:100]}'
                    }
                    skip_this_batch = True
            
            # Process the loss
            if not loss_dict.get('skip_batch', False):
                total_loss = loss_dict['loss']
                # Check for NaN loss
                if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    # AMP backward pass
                    if use_amp and scaler is not None:
                        scaler.scale(total_loss).backward()
                        
                        # Calculate gradient norm before clipping (with scaling)
                        scaler.unscale_(optimizer)
                        grad_norm = 0.0
                        param_count = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                                param_count += param.numel()
                        grad_norm = grad_norm ** 0.5
                        
                        # Gradient clipping
                        grad_clip_norm = getattr(cfg.training, 'grad_clip_norm', 0) if cfg else 0
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Regular backward pass (FP32)
                        total_loss.backward()
                        
                        # Calculate gradient norm
                        grad_norm = 0.0
                        param_count = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                                param_count += param.numel()
                        grad_norm = grad_norm ** 0.5
                        
                        # Gradient clipping
                        grad_clip_norm = getattr(cfg.training, 'grad_clip_norm', 0) if cfg else 0
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        
                        optimizer.step()
                    
                    # Store gradient norm for logging
                    loss_dict['grad_norm'] = grad_norm
                    loss_dict['param_count'] = param_count
                    processed_batches += 1
                    
                    # Accumulate losses for logging
                    for k, v in loss_dict.items():
                        if k in ['global_batch_size', 'local_batch_size', 'skip_batch']:
                            continue
                        v = v.item() if torch.is_tensor(v) else float(v)
                        loss_sums[k] += v
                        loss_counts[k] += 1
                    
                    # Track accuracy
                    acc = loss_dict.get("accuracy", None)
                    if acc is not None:
                        train_metrics.append(acc)
                else:
                    skipped_batches += 1
                    logging.warning(f"Rank {rank}: Batch {batch_count} - Skipped due to NaN/Inf loss (loss={total_loss.item() if torch.is_tensor(total_loss) else total_loss})")
            else:
                skipped_batches += 1
                skip_reason = loss_dict.get('skip_reason', 'Unknown')
                logging.warning(f"Rank {rank}: Batch {batch_count} - Skipped. Reason: {skip_reason}")
            
            # Increment global step counter for successful batches
            if not loss_dict.get('skip_batch', False):
                global_step += 1
                
                # Step-based logging
                if use_step_logging and global_step % log_every_k_steps == 0:
                    step_avg_losses = {f"train_{k}": loss_sums[k] / max(1, loss_counts[k]) for k in loss_sums}
                    step_accuracy = float(sum(train_metrics)) / max(1, len(train_metrics)) if train_metrics else 0.0
                    
                    step_logs = {}
                    step_logs.update(step_avg_losses)
                    step_logs.update({'train_accuracy': step_accuracy})
                    # Only add global_step for step-based logging (don't add epoch to avoid confusion)
                    step_logs.update({'global_step': global_step})
                    step_logs.update(extra_log_params)
                    
                    if should_log and log_fn:
                        if should_log:
                            # Build comprehensive console log message showing all loss components
                            main_loss = step_avg_losses.get('train_loss', 0)
                            log_msg = f"📊 Step {global_step}: Loss={main_loss:.4f}, Acc={step_accuracy:.4f}"
                            
                            # Add important loss components to console output
                            important_components = []
                            for k, v in step_avg_losses.items():
                                if k in ['train_loss', 'train_accuracy', 'train_grad_norm', 'train_param_count']:
                                    continue  # Skip already shown or not important for console
                                # Show component losses and accuracies
                                if '_loss' in k or '_accuracy' in k or 'violations' in k or 'margin' in k:
                                    clean_name = k.replace('train_', '').replace('_loss', '').replace('_accuracy', '_acc')
                                    important_components.append(f"{clean_name}={v:.4f}")
                            
                            if important_components:
                                log_msg += f", {', '.join(important_components[:5])}"  # Show top 5 components
                            
                            logging.info(log_msg)
                        # Pass step to wandb for proper timeline synchronization
                        try:
                            log_fn(step_logs, step=global_step)
                        except TypeError:
                            # Fallback for log functions that don't accept step parameter
                            log_fn(step_logs)
                
                # Step-based evaluation
                if use_step_evaluation and global_step % eval_every_k_steps == 0:
                    if should_log:
                        logging.info(f"🔍 Running step-based evaluation at step {global_step}")
                    
                    if evaluate_fn is not None:
                        base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                        eval_logs = evaluate_fn(base_model, val_loader, device, loss_fn, **loss_kwargs)
                        # Only add global_step for step-based logging (don't add epoch to avoid confusion)
                        eval_logs.update({'global_step': global_step})
                        
                        # Step-based dataset evaluation
                        benchmark_results = {}
                        if dataset_evaluator is not None:
                            try:
                                if should_log:
                                    logging.info(f"🔍 Starting dataset evaluation at step {global_step}")
                                import clip
                                model_name = getattr(cfg.model, 'model_name', "ViT-B/32") if cfg else "ViT-B/32"
                                _, preprocess = clip.load(model_name, device=device)
                                benchmark_results = dataset_evaluator.evaluate_all(
                                    model=base_model,
                                    clip_model=clip_model,
                                    preprocess=preprocess,
                                    alignment_type=alignment_type,
                                    step=global_step,
                                    epoch=None,  # Don't pass epoch for step-based evaluation
                                    wandb_log=False,
                                    is_initial_eval=False
                                )
                                if should_log:
                                    logging.info(f"✅ Step {global_step}: Dataset evaluation completed with {len(benchmark_results)} results")
                            except Exception as e:
                                if should_log:
                                    logging.error(f"❌ Step {global_step}: Dataset evaluation failed: {e}")
                                benchmark_results = {}
                        
                        # Step-based best model tracking
                        if metric_key in eval_logs and eval_logs[metric_key] is not None:
                            if eval_logs[metric_key] > best_metric_value:
                                best_metric_value = eval_logs[metric_key]
                                best_epoch = epoch
                                best_step = global_step
                                base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                                best_model_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                                if should_log:
                                    logging.info(f"🎯 New best model at step {global_step}: {metric_key}={best_metric_value:.4f}")
                        
                        # Log step-based evaluation results
                        eval_logs.update(benchmark_results)
                        if should_log and log_fn:
                            # Pass step to wandb for proper timeline synchronization
                            try:
                                log_fn(eval_logs, step=global_step)
                            except TypeError:
                                # Fallback for log functions that don't accept step parameter
                                log_fn(eval_logs)
                
                # Step-based checkpoint saving
                if use_step_saving and global_step % save_every_k_steps == 0:
                    save_path = getattr(cfg.training, 'save_path', None) if cfg else None
                    exp_name = getattr(cfg, 'exp_name', 'experiment') if cfg else 'experiment'
                    if should_log and save_path:
                        ckpt_dir = os.path.join(save_path, f"{exp_name}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                        step_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pt")
                        torch.save(base_model.state_dict(), step_ckpt_path)
                        if should_log:
                            logging.info(f"💾 Saved checkpoint at step {global_step}: {step_ckpt_path}")
                
                # Step-based sample logging (visualizations)
                if use_step_sample_logging and global_step % sample_every_k_steps == 0:
                    if sample_logger is not None and should_log:
                        try:
                            if should_log:
                                logging.info(f"🎨 Generating sample visualizations at step {global_step}")
                            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                            step_sample_data = sample_logger.log_epoch_samples(
                                model=base_model,
                                epoch=epoch + 1,
                                wandb_prefix="train_samples",  # Use consistent prefix for evolution tracking
                                log_images=True,
                                log_metrics=True
                            )
                            
                            if step_sample_data and log_fn:
                                # Create a separate log entry for step-based samples
                                # Only use global_step for step-based logging (don't add epoch)
                                sample_logs = {'global_step': global_step}
                                
                                # Add sample metrics with prefix
                                if 'metrics' in step_sample_data:
                                    prefix = step_sample_data.get('wandb_prefix', 'train_samples')
                                    for k, v in step_sample_data['metrics'].items():
                                        sample_logs[f"{prefix}/{k}"] = v
                                
                                # Add sample images
                                if 'images' in step_sample_data:
                                    prefix = step_sample_data.get('wandb_prefix', 'train_samples')
                                    sample_logs[f"{prefix}/sample_panels"] = step_sample_data['images']
                                
                                # Log to wandb
                                try:
                                    log_fn(sample_logs, step=global_step)
                                except TypeError:
                                    log_fn(sample_logs)
                                
                                if should_log:
                                    logging.info(f"✅ Step {global_step}: Sample visualizations logged")
                        except Exception as e:
                            if should_log:
                                logging.error(f"❌ Step {global_step}: Sample logging failed: {e}")
                                import traceback
                                logging.error(f"📍 Step sample logging traceback:\n{traceback.format_exc()}")
            
            if scheduler is not None and scheduler_on == 'step' and not skip_this_batch:
                scheduler.step()
            
            # Update progress bar
            if should_log and pbar is not None:
                postfix_dict = {}
                main_loss = 0.0  # Default value for skipped batches
                
                if loss_dict and not loss_dict.get('skip_batch', False):
                    # Add main metrics with cleaner formatting
                    main_loss = loss_dict.get('total_loss', loss_dict.get('loss', 0))
                    if isinstance(main_loss, torch.Tensor):
                        main_loss = main_loss.item()
                    postfix_dict['Loss'] = f"{main_loss:.3f}"
                    
                    if 'accuracy' in loss_dict:
                        acc = loss_dict['accuracy']
                        if isinstance(acc, torch.Tensor):
                            acc = acc.item()
                        postfix_dict['Acc'] = f"{acc:.3f}"
                    
                    # Add all loss components (not just main loss)
                    for k, v in loss_dict.items():
                        # Skip non-loss metrics
                        if k in ['accuracy', 'global_batch_size', 'learning_rate', 'temperature', 'grad_norm']:
                            continue
                        # Skip the main loss (already displayed)
                        if k in ['loss', 'total_loss']:
                            continue
                        # Add individual loss components to postfix
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        # Format key nicely for display
                        display_key = k.replace('_loss', '').replace('train_', '').title()[:8]
                        postfix_dict[display_key] = f"{v:.3f}"
                    
                    # Add batch size info
                    if 'global_batch_size' in loss_dict:
                        postfix_dict['BS'] = int(loss_dict['global_batch_size'])
                    
                    # Add learning rate and temperature from loss_dict
                    if 'learning_rate' in loss_dict:
                        lr = loss_dict['learning_rate']
                        postfix_dict['LR'] = f"{lr:.2e}"

                    if 'temperature' in loss_dict:
                        temp = loss_dict['temperature']
                        if isinstance(temp, torch.Tensor):
                            temp = temp.item()
                        postfix_dict['Temp'] = f"{temp:.3f}"
                    
                    # Add gradient norm if available
                    if 'grad_norm' in loss_dict:
                        grad_norm = loss_dict['grad_norm']
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm = grad_norm.item()
                        postfix_dict['GradNorm'] = f"{grad_norm:.2f}"
                        
                else:
                    # Show skip status when batch is skipped
                    postfix_dict['Status'] = 'SKIPPED'
                    
                # Update both description and postfix with step information
                batch_desc = f"Batch {batch_count}" + (f"/{total_batches}" if total_batches else "")
                step_desc = f" | Step {global_step}" if not loss_dict.get('skip_batch', False) else ""
                pbar.set_description(f"⚡ Epoch {epoch+1:3d}/{num_epochs} | {batch_desc}{step_desc} | Loss: {main_loss:.3f}")
                
                # Add step info to postfix if using step-based operations
                if use_step_logging or use_step_evaluation or use_step_saving:
                    postfix_dict['Step'] = global_step if not loss_dict.get('skip_batch', False) else f"{global_step}*"
                    
                pbar.set_postfix(postfix_dict)
                pbar.update(1)
        
        # Close progress bar
        if should_log and pbar is not None:
            pbar.close()
        
        # Log final batch count information
        if distributed:
            all_batch_counts = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
            local_count = torch.tensor([processed_batches], device=device, dtype=torch.long)
            dist.all_gather(all_batch_counts, local_count)
            
            if should_log:
                counts = [c.item() for c in all_batch_counts]
                logging.info(f"Epoch {epoch+1}: Processed batches per GPU: {counts}, Total: {sum(counts)}")
        
        # === Aggregate training metrics across GPUs ===
        if distributed:
            dist.barrier()
            
            # Aggregate losses
            for k in loss_sums:
                try:
                    sum_t = torch.tensor(loss_sums[k], device=device, dtype=torch.float32)
                    count_t = torch.tensor(loss_counts[k], device=device, dtype=torch.float32)
                    
                    dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
                    
                    loss_sums[k] = sum_t.item()
                    loss_counts[k] = max(1, int(count_t.item()))
                except Exception as e:
                    logging.warning(f"Metric aggregation failed for {k}: {e}")
            
            # Aggregate accuracy
            if train_metrics:
                try:
                    acc_sum = torch.tensor(sum(train_metrics), device=device, dtype=torch.float32)
                    acc_count = torch.tensor(len(train_metrics), device=device, dtype=torch.float32)
                    
                    dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)
                    
                    train_accuracy = acc_sum.item() / max(1, acc_count.item())
                except Exception as e:
                    logging.warning(f"Accuracy aggregation failed: {e}")
                    train_accuracy = 0.0
            else:
                train_accuracy = 0.0
        else:
            train_accuracy = float(sum(train_metrics)) / max(1, len(train_metrics)) if train_metrics else 0.0
        
        avg_losses = {f"train_{k}": loss_sums[k] / max(1, loss_counts[k]) for k in loss_sums}
        
        # === Validation (epoch-based, only if not using step-based evaluation) ===
        logs = {}
        benchmark_results = {}
        if evaluate_fn is not None and (epoch + 1) % evaluate_every_n == 0 and not use_step_evaluation:
            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
            logs = evaluate_fn(base_model, val_loader, device, loss_fn, **loss_kwargs)
            
            if dataset_evaluator is not None:
                try:
                    if should_log:
                        logging.info(f"🔍 Starting dataset evaluation for epoch {epoch + 1}")
                    import clip
                    model_name = getattr(cfg.model, 'model_name', "ViT-B/32") if cfg else "ViT-B/32"
                    _, preprocess = clip.load(model_name, device=device)
                    benchmark_results = dataset_evaluator.evaluate_all(
                        model=base_model,
                        clip_model=clip_model,
                        preprocess=preprocess,
                        alignment_type=alignment_type,
                        step=epoch + 1,
                        epoch=epoch + 1,
                        wandb_log=False,
                        is_initial_eval=False
                    )
                    if should_log:
                        logging.info(f"✅ Epoch {epoch + 1}: Dataset evaluation completed with {len(benchmark_results)} results")
                        
                        # Log the actual benchmark results for debugging
                        if benchmark_results:
                            logging.info("📊 Benchmark results:")
                            for key, value in benchmark_results.items():
                                if isinstance(value, (int, float)):
                                    logging.info(f"   {key}: {value:.4f}")
                                else:
                                    logging.info(f"   {key}: {value}")
                        else:
                            logging.warning("⚠️ Benchmark evaluation returned empty results")
                        
                except Exception as e:
                    if should_log:
                        error_msg = f"Dataset evaluation failed at epoch {epoch + 1}: {e}"
                        logging.error(f"❌ {error_msg}")
                        import traceback
                        logging.error(f"📍 Dataset evaluation traceback:\n{traceback.format_exc()}")
                    benchmark_results = {}
        
        # === Sample logging (track model learning on fixed samples) - epoch-based ===
        # Only run epoch-based sample logging if not using step-based sample logging
        sample_logger_data = {}
        if sample_logger is not None and should_log and not use_step_sample_logging:
            try:
                base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                sample_logger_data = sample_logger.log_epoch_samples(
                    model=base_model,
                    epoch=epoch + 1,
                    wandb_prefix="train_samples",
                    log_images=True,
                    log_metrics=True
                )
                if should_log and sample_logger_data:
                    logging.info(f"✅ Prepared sample visualizations for epoch {epoch + 1}")
            except Exception as e:
                if should_log:
                    logging.error(f"Sample logging failed at epoch {epoch + 1}: {e}")
                    import traceback
                    logging.error(f"📍 Sample logging traceback:\n{traceback.format_exc()}")
        
        # === Best model tracking (epoch-based, only if not using step-based evaluation) ===
        if not use_step_evaluation and metric_key in logs and logs[metric_key] is not None:
            if logs[metric_key] > best_metric_value:
                best_metric_value = logs[metric_key]
                best_epoch = epoch
                best_step = global_step  # Track the step even for epoch-based evaluation
                base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                best_model_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        
        # === Logging ===
        if should_log:
            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
            parameters_dict = {}
            
            if hasattr(base_model, "get_alphas"):
                alphas = base_model.get_alphas()
                if alphas.get('image') is not None:
                    parameters_dict.update({f'parameters/alpha_image_{i}': a.item() for i, a in enumerate(alphas['image'])})
                if alphas.get('text') is not None:
                    parameters_dict.update({f'parameters/alpha_text_{i}': a.item() for i, a in enumerate(alphas['text'])})
            
            if hasattr(base_model, "temperature"):
                temperature = getattr(base_model, "temperature", None)
                if temperature is not None:
                    parameters_dict['parameters/temperature'] = temperature.item() if torch.is_tensor(temperature) else temperature
                else:
                    # LabCLIP doesn't use temperature
                    parameters_dict['parameters/temperature'] = "N/A (LabCLIP)"

                       
            parameters_dict['parameters/lr'] = optimizer.param_groups[0]['lr']
            parameters_dict['train/processed_batches'] = processed_batches
            parameters_dict['train/skipped_batches'] = skipped_batches
            
            # Add gradient norm to logging if available
            if 'grad_norm' in avg_losses:
                parameters_dict['train/grad_norm'] = avg_losses['grad_norm']
            if 'param_count' in avg_losses:
                parameters_dict['train/param_count'] = avg_losses['param_count']
            
            logs.update(parameters_dict)
            logs.update(avg_losses)
            logs.update({'train_accuracy': train_accuracy})
            logs.update(extra_log_params)
            logs.update(benchmark_results)
            
            # Add sample logger metrics to the same log call
            if sample_logger_data:
                # Add sample metrics with prefix
                if 'metrics' in sample_logger_data:
                    prefix = sample_logger_data.get('wandb_prefix', 'train_samples')
                    for k, v in sample_logger_data['metrics'].items():
                        logs[f"{prefix}/{k}"] = v
                
                # Add sample images
                if 'images' in sample_logger_data:
                    prefix = sample_logger_data.get('wandb_prefix', 'train_samples')
                    logs[f"{prefix}/sample_panels"] = sample_logger_data['images']
            
            if log_fn:
                # Pass current global step to wandb for proper timeline synchronization
                try:
                    log_fn(logs, step=global_step)
                except TypeError:
                    # Fallback for log functions that don't accept step parameter
                    log_fn(logs)
            
            if isinstance(epoch_iter, tqdm):
                # Build comprehensive epoch summary for merged training
                epoch_metrics = {}
                
                # Add main losses
                main_loss = avg_losses.get('total_loss', avg_losses.get('loss', 0))
                epoch_metrics["Loss"] = f"{main_loss:.4f}"
                epoch_metrics["Acc"] = f"{train_accuracy:.4f}"
                
                # Add batch processing stats
                epoch_metrics["Proc"] = f"{processed_batches}"
                if skipped_batches > 0:
                    epoch_metrics["Skip"] = f"{skipped_batches}"
                
                # Add all loss component breakdown (not just 'aux' losses)
                loss_components = {}
                for k, v in avg_losses.items():
                    # Skip these meta keys
                    if k in ['total_loss', 'train_loss', 'loss', 'train_accuracy', 
                            'grad_norm', 'param_count', 'global_batch_size', 
                            'learning_rate', 'temperature']:
                        continue
                    # This is a loss component - add it
                    if '_loss' in k or k.endswith('_accuracy') or k in ['violations', 'avg_margin']:
                        loss_components[k] = v
                
                # Display individual loss components in epoch summary
                if loss_components:
                    for k, v in sorted(loss_components.items())[:5]:  # Show top 5 components
                        display_key = k.replace('train_', '').replace('_loss', '').title()[:8]
                        epoch_metrics[display_key] = f"{v:.3f}"

                # Add temperature
                if hasattr(base_model, "temperature"):
                    temperature = getattr(base_model, "temperature", None)
                    if temperature is not None:
                        epoch_metrics["T"] = f"{temperature.item():.2f}" if torch.is_tensor(temperature) else f"{temperature:.2f}"
                    else:
                        epoch_metrics["T"] = "N/A"

                # Add learning rate
                epoch_metrics["LR"] = f"{optimizer.param_groups[0]['lr']:.1e}"
                
                epoch_iter.set_postfix(epoch_metrics)
        
        if scheduler is not None and scheduler_on == 'epoch':
            scheduler.step()
        
        # === Save checkpoint ===
        save_path = getattr(cfg.training, 'save_path', None) if cfg else None
        exp_name = getattr(cfg, 'exp_name', 'experiment') if cfg else 'experiment'
        if should_log and save_path:
            ckpt_dir = os.path.join(save_path, f"{exp_name}")
            os.makedirs(ckpt_dir, exist_ok=True)
            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
            torch.save(base_model.state_dict(), os.path.join(ckpt_dir, "last_checkpoint.pt"))
    
    return {
        "best_model_state_dict": best_model_state,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_metric_value": best_metric_value,
        "metric_key": metric_key,
        "final_global_step": global_step,
    }




def unpack_neg_multilayer(batch, model, device):
    image_emb_dict, text_emb_dict, neg_text_emb_dict, _ = batch
    
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    
    image_emb = base_model.encode_image({k: v.to(device) for k, v in image_emb_dict.items()})
    text_emb  = base_model.encode_text({k: v.to(device) for k, v in text_emb_dict.items()})
    neg_text_emb = base_model.encode_text({k: v.to(device) for k, v in neg_text_emb_dict.items()})

    # Validate embeddings for NaN/Inf before returning
    if torch.isnan(image_emb).any() or torch.isinf(image_emb).any():
        logging.error("NaN/Inf detected in image embeddings during unpacking!")
        logging.error(f"Image emb stats: min={image_emb.min().item():.6f}, max={image_emb.max().item():.6f}")
        
    if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
        logging.error("NaN/Inf detected in text embeddings during unpacking!")
        logging.error(f"Text emb stats: min={text_emb.min().item():.6f}, max={text_emb.max().item():.6f}")
        
    if torch.isnan(neg_text_emb).any() or torch.isinf(neg_text_emb).any():
        logging.error("NaN/Inf detected in negative text embeddings during unpacking!")
        logging.error(f"Neg text emb stats: min={neg_text_emb.min().item():.6f}, max={neg_text_emb.max().item():.6f}")

    return {
        "image_embeddings": image_emb,  # [B, D] - no squeeze needed!
        "text_embeddings": text_emb,
        "neg_text_embeddings": neg_text_emb,
        "temperature": base_model.temperature.to(device),
        "device": device
    }

def unpack_ft_multilayer(batch, model, device):
    """
    Unpacking function that handles both standard mode (single positive) and multi-caption mode (multiple positives).
    
    NEW FORMAT: Each positive caption has a paired negative.
    - pos_tokens: [B, 1+N, 77] where index 0 = original, 1:N+1 = component captions
    - neg_tokens: [B, 1+N, 77] where neg[i] is paired with pos[i]
    - paraphrase_tokens: [B, 77] - paraphrase of original caption (for sentence alignment loss)
    - has_paraphrase: [B] - boolean mask for valid paraphrases
    - caption_valid_mask: [B, 1+N] boolean mask for valid pos-neg pairs
    
    Returns:
        dict with:
            - image_embeddings: [B, D]
            - text_embeddings: [B, 1+N, D] - positive caption embeddings
            - neg_text_embeddings: [B, 1+N, D] - paired negative embeddings (NEW: now matches positives shape!)
            - paraphrase_embeddings: Optional[torch.Tensor] [B, D] - paraphrase embeddings (None if no paraphrases)
            - has_paraphrase: Optional[torch.Tensor] [B] - mask for valid paraphrases
            - temperature: scalar
            - device: device
            - num_positive_captions: int (1 for standard mode, 1+N for multi-caption mode)
            - components_per_caption: Optional[torch.Tensor] [B, 1+N] - components in each caption
            - num_components_available: Optional[torch.Tensor] [B] - total components available for each sample
            - caption_valid_mask: Optional[torch.Tensor] [B, 1+N] - boolean mask for valid pairs
    """
    # Handle both dictionary format (from collate_fn) and tuple format (legacy)
    components_per_caption = None
    num_components_available = None
    caption_valid_mask = None
    paraphrase_tokens = None
    has_paraphrase = None
    
    if isinstance(batch, dict):
        # Dictionary format from collate_fn (e.g., LAION400MNeg with tar-grouped batching)
        images = batch["images"]
        captions = batch["pos_tokens"]  # [B, 1+N, 77]
        neg_captions = batch.get("neg_tokens", batch.get("neg_token"))  # [B, 1+N, 77] or legacy [B, 77]
        # Extract coverage and validity information
        components_per_caption = batch.get("components_per_caption")  # [B, 1+N]
        num_components_available = batch.get("num_components_available")  # [B]
        caption_valid_mask = batch.get("caption_valid_mask")  # [B, 1+N]
        # Extract paraphrase data
        paraphrase_tokens = batch.get("paraphrase_tokens")  # [B, 77] or None
        has_paraphrase = batch.get("has_paraphrase")  # [B] or None
    else:
        # Tuple format (legacy)
        if len(batch) == 4:
            images, captions, neg_captions, _ = batch
        else:
            images, captions, neg_captions = batch
        
    images = images.to(device)
    captions = captions.to(device)  # [B, 1+N, 77]
    neg_captions = neg_captions.to(device)
    
    # Move coverage tensors to device if present
    if components_per_caption is not None:
        components_per_caption = components_per_caption.to(device)
    if num_components_available is not None:
        num_components_available = num_components_available.to(device)
    if caption_valid_mask is not None:
        caption_valid_mask = caption_valid_mask.to(device)
    
    # Move paraphrase tensors to device if present
    if paraphrase_tokens is not None:
        paraphrase_tokens = paraphrase_tokens.to(device)
    if has_paraphrase is not None:
        has_paraphrase = has_paraphrase.to(device)
    
    # Detect mode: check if captions has multiple positives per sample
    batch_size = images.shape[0]
    num_positive_captions = captions.shape[1] if captions.dim() == 3 else 1
    
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    
    # Encode images (same for both modes)
    image_emb = base_model.encode_image(images)  # [B, D]
    
    # Encode positive text captions
    if num_positive_captions > 1:
        # Multi-caption mode: encode all positive captions
        # Reshape to [B*(1+N), 77] for batch encoding
        captions_flat = captions.view(-1, captions.shape[-1])  # [B*(1+N), 77]
        text_emb_flat = base_model.encode_text(captions_flat)  # [B*(1+N), D]
        # Reshape back to [B, 1+N, D]
        text_emb = text_emb_flat.view(batch_size, num_positive_captions, -1)  # [B, 1+N, D]
    else:
        # Standard mode: single positive caption per sample
        if captions.dim() == 3:
            captions = captions.squeeze(1)  # [B, 77]
        text_emb = base_model.encode_text(captions)  # [B, D]
        # Add dimension for consistency: [B, 1, D]
        text_emb = text_emb.unsqueeze(1)  # [B, 1, D]
    
    # Encode negative captions - NEW: handle paired negatives format [B, 1+N, 77]
    if neg_captions.dim() == 3 and neg_captions.shape[1] == num_positive_captions:
        # NEW FORMAT: Paired negatives [B, 1+N, 77] - one negative per positive
        neg_captions_flat = neg_captions.view(-1, neg_captions.shape[-1])  # [B*(1+N), 77]
        neg_text_emb_flat = base_model.encode_text(neg_captions_flat)  # [B*(1+N), D]
        # Reshape back to [B, 1+N, D]
        neg_text_emb = neg_text_emb_flat.view(batch_size, num_positive_captions, -1)  # [B, 1+N, D]
    elif neg_captions.dim() == 2:
        # LEGACY FORMAT: Single negative per sample [B, 77]
        neg_text_emb = base_model.encode_text(neg_captions)  # [B, D]
        # Broadcast to match positive shape: [B, 1, D] -> will be [B, 1+N, D] if needed
        neg_text_emb = neg_text_emb.unsqueeze(1)  # [B, 1, D]
        if num_positive_captions > 1:
            # Broadcast single negative to all positive captions
            # Note: .contiguous() is needed after expand for distributed all_gather
            neg_text_emb = neg_text_emb.expand(-1, num_positive_captions, -1).contiguous()  # [B, 1+N, D]
    else:
        # Unexpected format - log warning and try to handle
        logging.warning(f"Unexpected neg_captions shape: {neg_captions.shape}, attempting to handle...")
        if neg_captions.dim() > 2:
            # Take first negative if multiple
            neg_captions = neg_captions[:, 0, :]
        neg_text_emb = base_model.encode_text(neg_captions)  # [B, D]
        neg_text_emb = neg_text_emb.unsqueeze(1)  # [B, 1, D]
        if num_positive_captions > 1:
            # Note: .contiguous() is needed after expand for distributed all_gather
            neg_text_emb = neg_text_emb.expand(-1, num_positive_captions, -1).contiguous()

    # Encode paraphrase captions if available (for sentence alignment loss)
    paraphrase_emb = None
    if paraphrase_tokens is not None and has_paraphrase is not None:
        # Only encode if at least some samples have paraphrases
        if has_paraphrase.any():
            paraphrase_emb = base_model.encode_text(paraphrase_tokens)  # [B, D]
        # Note: samples without paraphrases will be masked out using has_paraphrase

    return {
        "image_embeddings": image_emb,  # [B, D]
        "text_embeddings": text_emb,  # [B, 1+N, D]
        "neg_text_embeddings": neg_text_emb,  # [B, 1+N, D] - NOW MATCHES POSITIVES!
        "paraphrase_embeddings": paraphrase_emb,  # [B, D] or None - paraphrase embeddings
        "has_paraphrase": has_paraphrase,  # [B] or None - mask for valid paraphrases
        "temperature": base_model.temperature.to(device),
        "device": device,
        "num_positive_captions": num_positive_captions,
        "components_per_caption": components_per_caption,  # [B, 1+N] or None
        "num_components_available": num_components_available,  # [B] or None
        "caption_valid_mask": caption_valid_mask,  # [B, 1+N] or None - NEW!
    }


def unpack_ft_tqa(batch, model, device):
    """
    Unpacking function for TextQueryAggregator (TQA) head.
    
    TQA uses cross-attention: v_cls' = v_cls + Attn(q=text, k=patches, v=patches)
    This requires passing text to encode_image() so the cross-attention fires.
    
    For training, we condition the image on the FIRST positive caption (original/full caption).
    This creates caption-conditioned image embeddings that should align better with text.
    
    Args:
        batch: Batch from dataloader (dict or tuple format)
        model: Model (may be DDP-wrapped)
        device: Target device
        
    Returns:
        dict with:
            - image_embeddings: [B, D] - caption-conditioned image embeddings
            - text_embeddings: [B, 1+N, D] - positive caption embeddings
            - neg_text_embeddings: [B, 1+N, D] - paired negative embeddings
            - temperature: scalar
            - device: device
            - num_positive_captions: int (1 for standard mode, 1+N for multi-caption mode)
            - components_per_caption: Optional[torch.Tensor] [B, 1+N]
            - num_components_available: Optional[torch.Tensor] [B]
            - caption_valid_mask: Optional[torch.Tensor] [B, 1+N]
    """
    # Handle both dictionary format (from collate_fn) and tuple format (legacy)
    components_per_caption = None
    num_components_available = None
    caption_valid_mask = None
    
    if isinstance(batch, dict):
        # Dictionary format from collate_fn
        images = batch["images"]
        captions = batch["pos_tokens"]  # [B, 1+N, 77]
        neg_captions = batch.get("neg_tokens", batch.get("neg_token"))
        components_per_caption = batch.get("components_per_caption")
        num_components_available = batch.get("num_components_available")
        caption_valid_mask = batch.get("caption_valid_mask")
    else:
        # Tuple format (legacy)
        if len(batch) == 4:
            images, captions, neg_captions, _ = batch
        else:
            images, captions, neg_captions = batch
        
    images = images.to(device)
    captions = captions.to(device)
    neg_captions = neg_captions.to(device)
    
    # Move coverage tensors to device if present
    if components_per_caption is not None:
        components_per_caption = components_per_caption.to(device)
    if num_components_available is not None:
        num_components_available = num_components_available.to(device)
    if caption_valid_mask is not None:
        caption_valid_mask = caption_valid_mask.to(device)
    
    # Detect mode: check if captions has multiple positives per sample
    batch_size = images.shape[0]
    num_positive_captions = captions.shape[1] if captions.dim() == 3 else 1
    
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    
    # ============================================================
    # KEY DIFFERENCE FROM unpack_ft_multilayer:
    # Encode text FIRST, then use it to condition image encoding
    # ============================================================
    
    # Encode positive text captions
    if num_positive_captions > 1:
        # Multi-caption mode: encode all positive captions
        captions_flat = captions.view(-1, captions.shape[-1])  # [B*(1+N), 77]
        text_emb_flat = base_model.encode_text(captions_flat)  # [B*(1+N), D]
        text_emb = text_emb_flat.view(batch_size, num_positive_captions, -1)  # [B, 1+N, D]
    else:
        # Standard mode
        if captions.dim() == 3:
            captions = captions.squeeze(1)  # [B, 77]
        text_emb = base_model.encode_text(captions)  # [B, D]
        text_emb = text_emb.unsqueeze(1)  # [B, 1, D]
    
    # Extract the first caption tokens for image conditioning
    # (the original/full caption, not component captions)
    if captions.dim() == 3:
        conditioning_tokens = captions[:, 0, :]  # [B, 77] - first caption
    else:
        conditioning_tokens = captions  # [B, 77]
    
    # Encode image WITH text conditioning (this triggers TQA cross-attention!)
    # TextQueryAggregatorPipeline.encode_image(image, text=...) accepts optional text
    image_emb = base_model.encode_image(images, text=conditioning_tokens)  # [B, D]
    
    # Encode negative captions
    if neg_captions.dim() == 3 and neg_captions.shape[1] == num_positive_captions:
        # Paired negatives [B, 1+N, 77]
        neg_captions_flat = neg_captions.view(-1, neg_captions.shape[-1])
        neg_text_emb_flat = base_model.encode_text(neg_captions_flat)
        neg_text_emb = neg_text_emb_flat.view(batch_size, num_positive_captions, -1)
    elif neg_captions.dim() == 2:
        # Single negative per sample [B, 77]
        neg_text_emb = base_model.encode_text(neg_captions)
        neg_text_emb = neg_text_emb.unsqueeze(1)
        if num_positive_captions > 1:
            neg_text_emb = neg_text_emb.expand(-1, num_positive_captions, -1).contiguous()
    else:
        # Fallback
        logging.warning(f"Unexpected neg_captions shape in TQA unpack: {neg_captions.shape}")
        if neg_captions.dim() > 2:
            neg_captions = neg_captions[:, 0, :]
        neg_text_emb = base_model.encode_text(neg_captions)
        neg_text_emb = neg_text_emb.unsqueeze(1)
        if num_positive_captions > 1:
            neg_text_emb = neg_text_emb.expand(-1, num_positive_captions, -1).contiguous()

    return {
        "image_embeddings": image_emb,  # [B, D] - CAPTION-CONDITIONED!
        "text_embeddings": text_emb,  # [B, 1+N, D]
        "neg_text_embeddings": neg_text_emb,  # [B, 1+N, D]
        "temperature": base_model.temperature.to(device),
        "device": device,
        "num_positive_captions": num_positive_captions,
        "components_per_caption": components_per_caption,
        "num_components_available": num_components_available,
        "caption_valid_mask": caption_valid_mask,
    }
