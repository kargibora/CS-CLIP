import argparse
import logging
import math
import os
import random
import json
import types
from collections import defaultdict
import numpy as np
import clip
import torch
import wandb
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR, StepLR

from data_loading import (
    MultiLayerNegEmbeddingsDataset, LazyMultiLayerNegEmbeddingsDataset,
    TarGroupedMultiLayerNegEmbeddingsDataset,
)

from utils.optimizer_factory import make_optimizer_and_scheduler as make_modular_optimizer_and_scheduler


from utils.dist import (
    is_main_process,
    create_distributed_dataloader,
    get_world_size,
    get_rank,
)

from utils.cache import (
    is_lazy_embedding_loader,
)


def _filter_required_layers(emb_store, required_layers):
    """Keep only the layers you need when the store is a dict; pass-through for lazy loaders."""
    if emb_store is None:
        return emb_store
    if is_lazy_embedding_loader(emb_store):
        return emb_store
    return {k: v for k, v in emb_store.items() if k in required_layers}

def _pick_lazy_mode(image_embeddings, caption_embeddings) -> bool:
    """Decide if we should use lazy datasets."""
    return is_lazy_embedding_loader(image_embeddings) or is_lazy_embedding_loader(caption_embeddings)

def _build_train_val_datasets(
    use_lazy_loading: bool,
    image_embeddings,
    caption_embeddings,
    train_neg_indices_dict,
    val_neg_indices_dict,
    train_indices,
    val_indices,
    train_caption_indices,
    val_caption_indices,
    dataset=None,
    args=None,
    dtype=torch.float32,
):
    # Check if we should use tar-grouped datasets for LAION
    use_tar_grouped = (
        args is not None and 
        getattr(args, 'dataset', None) == 'LAION400M' and
        getattr(args, 'use_tar_batching', True) and
        dataset is not None and
        hasattr(dataset, 'get_available_tars')
    )
    
    if is_main_process():
        logging.info(f"Dataset creation - use_tar_grouped: {use_tar_grouped}")
        if dataset is not None:
            logging.info(f"  - Dataset type: {type(dataset)}")
            logging.info(f"  - Has get_available_tars: {hasattr(dataset, 'get_available_tars')}")
    
    if use_tar_grouped:
        # Debug: Get actual rank for logging
        rank = get_rank()
        logging.info(f"Rank {rank}: Creating tar-grouped datasets for sequential tar processing")
        logging.info(f"Rank {rank}: About to create train dataset")
        
        # Use cache_size=1 for sequential processing as discussed
        train_data = TarGroupedMultiLayerNegEmbeddingsDataset(
            image_embeddings=image_embeddings,
            text_embeddings=caption_embeddings,
            neg_indices_dict=train_neg_indices_dict,
            dataset=dataset,
            indices=train_indices,
            caption_indices=train_caption_indices,
            dtype=dtype,
            tar_cache_size=1  # Sequential processing - only current tar needed
        )
        
        logging.info(f"Rank {rank}: Train dataset created successfully")
        
        logging.info(f"Rank {rank}: About to create validation dataset")
        
        val_data = TarGroupedMultiLayerNegEmbeddingsDataset(
            image_embeddings=image_embeddings,
            text_embeddings=caption_embeddings,
            neg_indices_dict=val_neg_indices_dict,
            dataset=dataset,
            indices=val_indices,
            caption_indices=val_caption_indices,
            dtype=dtype,
            tar_cache_size=1  # Sequential processing - only current tar needed
        )
        
        logging.info(f"Rank {rank}: Validation dataset created successfully")
        
        if is_main_process():
            logging.info("✓ Created tar-grouped datasets")
            logging.info(f"  - Train available_tars: {len(train_data.available_tars)}")
            logging.info(f"  - Val available_tars: {len(val_data.available_tars)}")
            logging.info("  - Using tar_cache_size=1 for sequential processing")

    elif use_lazy_loading:
        train_data = LazyMultiLayerNegEmbeddingsDataset(
            image_embeddings=image_embeddings,
            text_embeddings=caption_embeddings,
            neg_indices_dict=train_neg_indices_dict,
            indices=train_indices,
            caption_indices=train_caption_indices,
            dtype=dtype,
        )
        val_data = LazyMultiLayerNegEmbeddingsDataset(
            image_embeddings=image_embeddings,
            text_embeddings=caption_embeddings,
            neg_indices_dict=val_neg_indices_dict,
            indices=val_indices,
            caption_indices=val_caption_indices,
            dtype=dtype,
        )
    else:
        train_data = MultiLayerNegEmbeddingsDataset(
            image_embeddings=image_embeddings,
            text_embeddings=caption_embeddings,
            neg_indices_dict=train_neg_indices_dict,
            indices=train_indices,
            caption_indices=train_caption_indices,
            dtype=dtype,
        )
        val_data = MultiLayerNegEmbeddingsDataset(
            image_embeddings=image_embeddings,
            text_embeddings=caption_embeddings,
            neg_indices_dict=val_neg_indices_dict,
            indices=val_indices,
            caption_indices=val_caption_indices,
            dtype=dtype,
        )
    
    return train_data, val_data

def _get_optimized_dataloader_params(dataset_name, base_num_workers=2):
    # Keep your original logic here
    return get_optimized_dataloader_params(dataset_name, base_num_workers)

def _build_dataloaders(args, train_data, val_data, is_distributed):
    # Debug: Log which processes reach dataloader creation
    rank = get_rank()
    logging.info(f"Rank {rank}: Starting dataloader creation")
    
    base_num_workers = getattr(args, 'num_workers', 2)
    params = _get_optimized_dataloader_params(args.dataset, base_num_workers)
    num_workers = params['num_workers']

    if args.dataset == 'LAION400M' and is_main_process():
        logging.info("Using optimized DataLoader settings for LAION400M:")
        logging.info(f"  - num_workers: {num_workers}")
        logging.info(f"  - persistent_workers: {params['persistent_workers']}")
        logging.info(f"  - prefetch_factor: {params['prefetch_factor']}")
        logging.info("  - Ensure ulimit -n is high (e.g., 65535)")
        
        # Debug the dataset type and attributes
        logging.info(f"Train dataset type: {type(train_data)}")
        logging.info(f"Has available_tars: {hasattr(train_data, 'available_tars')}")
        if hasattr(train_data, 'available_tars'):
            logging.info(f"Number of available tars: {len(train_data.available_tars)}")
        logging.info(f"use_tar_batching: {getattr(args, 'use_tar_batching', True)}")

    # Use tar-grouped batching for LAION400M when using TarGroupedMultiLayerNegEmbeddingsDataset
    tar_batching_conditions = [
        args.dataset == 'LAION400M',
        getattr(args, 'use_tar_batching', True),
        hasattr(train_data, 'available_tars')
    ]
    
    if is_main_process():
        logging.info(f"Tar batching conditions: {tar_batching_conditions}")
    
    if all(tar_batching_conditions):
        from data_loading.base import TarStreamingDataLoader
        
        if is_main_process():
            logging.info("✓ Using high-performance tar-streaming dataloader!")
            logging.info(f"  - Train tars: {len(train_data.available_tars)}")
            logging.info(f"  - Val tars: {len(val_data.available_tars)}")
            logging.info("  - Bulk tar loading with background prefetching")
            logging.info("  - Zero I/O wait between tars")
            logging.info("  - Each GPU processes complete tars sequentially")
        
        # Create optimized tar-streaming dataloaders
        train_dl = TarStreamingDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle_tars=True,
            shuffle_within_tar=True,
            drop_last=False,
            distributed=is_distributed,
            seed=args.seed if hasattr(args, 'seed') else 42,
            prefetch_next_tar=True,  # Enable background prefetching
            pin_memory=params['pin_memory'],
            debug_timing=True,  # Force enable debug timing to identify bottlenecks
        )
        
        val_dl = TarStreamingDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle_tars=False,
            shuffle_within_tar=False,
            drop_last=False,
            distributed=is_distributed,
            seed=args.seed if hasattr(args, 'seed') else 42,
            prefetch_next_tar=True,  # Enable background prefetching
            pin_memory=params['pin_memory'],
            debug_timing=False,  # Disable for validation to reduce noise
        )
        
        # For compatibility, create dummy samplers (not used by TarStreamingDataLoader)
        train_sampler = train_dl.batch_sampler
        val_sampler = val_dl.batch_sampler
        
        if is_main_process():
            logging.info("✓ Tar-streaming dataloaders created successfully")
            logging.info("  - Background tar prefetching enabled")
            logging.info("  - Bulk embedding loading per tar")
            logging.info("  - Optimized I/O and memory usage")
        
        # Debug: Log completion for all ranks
        logging.info(f"Rank {get_rank()}: Tar-streaming dataloaders created successfully")
            
    else:
        if is_main_process():
            logging.info("✗ Using standard dataloaders (not tar-grouped)")
            logging.info(f"  - Conditions failed: dataset={args.dataset}, tar_batching={getattr(args, 'use_tar_batching', True)}, has_tars={hasattr(train_data, 'available_tars')}")
        
        # Standard distributed dataloaders
        train_dl, train_sampler = create_distributed_dataloader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            distributed=is_distributed,
            pin_memory=params['pin_memory'],
        )
        val_dl, val_sampler = create_distributed_dataloader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            distributed=is_distributed,
            pin_memory=params['pin_memory'],
        )
    
    # Debug: Log completion for all ranks
    logging.info(f"Rank {get_rank()}: Dataloader creation completed successfully")
    
    return train_dl, val_dl, train_sampler, val_sampler

def infer_layer_layout(emb_store):
    """
    Return (layer_names, layer_dims) for lazy or regular stores.
    """
    if is_lazy_embedding_loader(emb_store):
        if hasattr(emb_store, 'metadata'):
            names = emb_store.layer_names
            dims = [emb_store.metadata['embedding_dims'][ln] for ln in names]
            return names, dims
        # Fallback: probe one row
        names = emb_store.layer_names
        probe = emb_store.get_embeddings(names[0], [0])
        dims = [probe.shape[-1] for _ in names]
        return names, dims
    else:
        names = list(emb_store.keys())
        dims = [emb_store[ln].shape[-1] for ln in names]
        return names, dims

def sync_model_config_across_ranks(image_layer_names, text_layer_names, image_layer_dims, text_layer_dims, device):
    if not (dist.is_available() and dist.is_initialized()):
        return image_layer_names, text_layer_names, image_layer_dims, text_layer_dims

    rank = dist.get_rank()
    logging.info(f"Rank {rank}: About to enter first barrier in sync_model_config_across_ranks")
    
    dist.barrier()
    
    logging.info(f"Rank {rank}: Passed first barrier, preparing payload")
    
    payload = {
        'image_layer_names': image_layer_names,
        'text_layer_names': text_layer_names,
        'image_layer_dims': image_layer_dims,
        'text_layer_dims': text_layer_dims,
    } if dist.get_rank() == 0 else None

    logging.info(f"Rank {rank}: Payload prepared, about to broadcast")

    try:
        if hasattr(dist, 'broadcast_object_list'):
            logging.info(f"Rank {rank}: Using broadcast_object_list")
            lst = [payload] if dist.get_rank() == 0 else [None]
            dist.broadcast_object_list(lst, src=0)
            if dist.get_rank() != 0:
                payload = lst[0]
            logging.info(f"Rank {rank}: broadcast_object_list completed")
        else:
            logging.info(f"Rank {rank}: Using manual pickle broadcast")
            import pickle
            if dist.get_rank() == 0:
                buf = pickle.dumps(payload)
                size = torch.tensor([len(buf)], dtype=torch.long, device=device)
            else:
                size = torch.tensor([0], dtype=torch.long, device=device)
            
            logging.info(f"Rank {rank}: About to broadcast size")
            dist.broadcast(size, src=0)
            logging.info(f"Rank {rank}: Size broadcast completed")
            
            if dist.get_rank() == 0:
                t = torch.frombuffer(memoryview(buf), dtype=torch.uint8).to(device)
            else:
                t = torch.empty(size.item(), dtype=torch.uint8, device=device)
            
            logging.info(f"Rank {rank}: About to broadcast tensor")
            dist.broadcast(t, src=0)
            logging.info(f"Rank {rank}: Tensor broadcast completed")
            
            if dist.get_rank() != 0:
                payload = pickle.loads(bytes(t.cpu().tolist()))
    except Exception as e:
        logging.warning(f"Config broadcast failed on rank {dist.get_rank()}: {e}")
        return image_layer_names, text_layer_names, image_layer_dims, text_layer_dims

    logging.info(f"Rank {rank}: About to enter final barrier in sync_model_config_across_ranks")
    dist.barrier()
    logging.info(f"Rank {rank}: sync_model_config_across_ranks completed successfully")
    
    return (payload['image_layer_names'], payload['text_layer_names'],
            payload['image_layer_dims'], payload['text_layer_dims'])

def sync_ddp_params(model_wrapper):
    if not (dist.is_available() and dist.is_initialized()):
        return
    base = model_wrapper.get_base_model()
    for p in base.parameters():
        dist.broadcast(p.data, src=0)
    if hasattr(base, 't') and base.t is not None:
        dist.broadcast(base.t.data, src=0)
    dist.barrier()
    if is_main_process():
        logging.info("✓ Model parameters synchronized across all ranks")

def make_optimizer_and_scheduler(model, cfg, train_dataloader):
    # Use the modular factory for optimizer and scheduler creation
    return make_modular_optimizer_and_scheduler(model, cfg, train_dataloader)

def get_optimized_dataloader_params(dataset_name, base_num_workers=2):
    """Get optimized DataLoader parameters for different datasets, especially tar-based ones."""
    if dataset_name == 'LAION400M':
        # Optimized settings for large-scale tar-based LAION dataset
        return {
            'num_workers': max(4, base_num_workers),  # Ensure minimum workers for standard mode
            'persistent_workers': True,
            'prefetch_factor': 4,  # Reduced from 8 to save memory
            'pin_memory': True,
        }
    else:
        # Standard settings for other datasets
        return {
            'num_workers': base_num_workers,
            'persistent_workers': False,
            'prefetch_factor': 8,
            'pin_memory': True,
        }
    

def set_scheduler(optimizer, epochs, train_dataloader, scheduler_type="none"):
    num_training_steps = epochs
    num_warmup_steps = int(num_training_steps * 0.05)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: float(step+1) / float(max(1, num_warmup_steps)))
    if scheduler_type == "linear":
        def linear_decay_lambda(step):
            if step < num_warmup_steps:
                return 1.0
            return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
        main_scheduler = LambdaLR(optimizer, lr_lambda=linear_decay_lambda)
    elif scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=0)
    elif scheduler_type == "step":
        main_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    return scheduler

