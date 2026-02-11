import logging
import os
import random
import json
import psutil
import gc
from omegaconf import DictConfig
import hydra
import numpy as np
import clip
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR, StepLR
import glob
from omegaconf import OmegaConf


from data_loading import (
    MultiLayerNegEmbeddingsDataset, LazyMultiLayerNegEmbeddingsDataset,
    build_dataset_from_args,
    get_dataset_embedding_class,
    get_dataset_cache_name
)
from data_loading.laion import create_tar_grouped_dataloader

from models import CLIPEndToEndPipeline, CLIPFeaturePipeline, TextQueryAggregatorPipeline

from alignment.learning_alignment import (
    train_model_multigpu_merged_batch,
)

from utils.head_init import create_model_from_config
from utils.clip_wrapper import load_clip_model, get_available_models

from utils.align import (
    extract_intermediate_features,
)
from utils.omega import (
    log_omegaconf_config,
    save_omegaconf_config,
    reconstruct_config_from_args,
)

from utils.debug import (
    debug_validate_embedding_size,
    debug_validate_embedding_order,
    debug_validate_caption_embedding_order,
    debug_dump_samples,
    debug_dump_dataloader_samples,
    debug_check_nan_embeddings,
    debug_validate_caption_vocab_count,
    debug_caption_locality,
)
from utils.dist import (
    add_multigpu_args,
    distributed_train_wrapper,
    is_main_process,
    create_distributed_dataloader,
    get_world_size,
    get_rank,
    MultiGPUWrapper,
    set_seed,
    monitor_nccl_health,
    safe_barrier,
)

from utils.cache import (
    load_embeddings,
    save_embeddings,
    is_lazy_embedding_loader,
    prepare_distributed_embeddings,
    load_embeddings_tar_based,
    load_embeddings_lazy,
    clean_up_cache,
    compute_image_embeddings_streaming,  
    compute_caption_embeddings_streaming,
)

from utils.labclip_helpers import (
    _filter_required_layers,
    _pick_lazy_mode,
    _build_train_val_datasets,
    infer_layer_layout,
    sync_model_config_across_ranks,
    get_optimized_dataloader_params,
    sync_ddp_params,
    _build_dataloaders,
    make_optimizer_and_scheduler
)

from utils.ft_helpers import (
    _prep_clip_for_alignment,
    _probe_layer_dims_from_batch,
    _build_ft_dataloaders,
    _scale_lr_for_ddp,
    _make_ft_optimizer_and_scheduler,
)

from utils.evaluate import flatten_dict





logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --------------------- Memory Monitoring utilities --------------------
def log_memory_usage(stage="Unknown", rank=None, force_gc=True):
    """
    Log comprehensive memory usage including RAM, GPU memory, and Python object counts.
    
    Args:
        stage: Description of current stage (e.g., "Before Training", "Epoch 1")
        rank: Process rank for distributed training (if None, will try to get it)
        force_gc: Whether to force garbage collection before measuring
    """
    try:
        if rank is None:
            try:
                rank = get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
        
        # Force garbage collection to get accurate memory usage
        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # System RAM usage
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        ram_used_gb = memory_info.rss / (1024**3)  # Convert bytes to GB
        ram_total_gb = system_memory.total / (1024**3)
        ram_percent = (memory_info.rss / system_memory.total) * 100
        
        # GPU memory usage
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            gpu_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            gpu_info = f", GPU: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved, {gpu_max_allocated:.2f}GB peak"
        
        # Python object counts (optional, can be expensive)
        obj_count = ""
        if force_gc:
            import sys
            obj_count = f", Objects: {len(gc.get_objects())} total"
        
        # Log comprehensive memory info
        logging.info(f"[MEMORY-{stage}] Rank {rank}: RAM: {ram_used_gb:.2f}GB/{ram_total_gb:.2f}GB ({ram_percent:.1f}%){gpu_info}{obj_count}")
        
        # Log warning if memory usage is high
        if ram_percent > 80:
            logging.warning(f"[MEMORY-WARNING] Rank {rank}: High RAM usage ({ram_percent:.1f}%) at {stage}")
        
        if torch.cuda.is_available() and gpu_reserved > 10:  # More than 10GB GPU memory
            logging.warning(f"[MEMORY-WARNING] Rank {rank}: High GPU memory usage ({gpu_reserved:.2f}GB) at {stage}")
            
        # Return memory stats for wandb logging
        return {
            "memory/ram_used_gb": ram_used_gb,
            "memory/ram_percent": ram_percent,
            "memory/gpu_allocated_gb": gpu_allocated if torch.cuda.is_available() else 0,
            "memory/gpu_reserved_gb": gpu_reserved if torch.cuda.is_available() else 0,
        }
        
    except Exception as e:
        logging.warning(f"Failed to log memory usage at {stage}: {e}")
        return {}

def log_memory_trend(memory_stats, stage_name):
    """Log memory trends over time for debugging memory leaks."""
    if not hasattr(log_memory_trend, 'history'):
        log_memory_trend.history = []
    
    memory_stats['stage'] = stage_name
    log_memory_trend.history.append(memory_stats)
    
    # Keep only last 10 measurements
    if len(log_memory_trend.history) > 10:
        log_memory_trend.history.pop(0)
    
    # Check for memory leaks (increasing trend)
    if len(log_memory_trend.history) >= 3:
        recent_ram = [m['memory/ram_used_gb'] for m in log_memory_trend.history[-3:]]
        if all(recent_ram[i] < recent_ram[i+1] for i in range(len(recent_ram)-1)):
            ram_increase = recent_ram[-1] - recent_ram[0]
            if ram_increase > 1.0:  # More than 1GB increase
                logging.warning(f"[MEMORY-LEAK-WARNING] RAM usage increased by {ram_increase:.2f}GB over last 3 measurements")

# --------------------- Debug utilities --------------------
def run_labclip(
        dataset, 
        image_embeddings, 
        caption_embeddings, 
        split_dict, 
        device, 
        cfg, 
        clip_model, 
        preprocess,
        loss_kwargs=None):
    
    # Extract config sections for cleaner access
    train_cfg = cfg.training
    eval_cfg = cfg.evaluation  
    align_cfg = cfg.alignment
    dataset_cfg = cfg.dataset
    dist_cfg = cfg.dist
    
    # Create minimal args for utility functions that haven't been converted yet
    import types
    args = types.SimpleNamespace()
    args.dataset = dataset_cfg.name
    args.use_tar_batching = getattr(dataset_cfg, 'use_tar_batching', False)
    args.batch_size = train_cfg.batch_size
    args.seed = train_cfg.seed
    args.num_workers = getattr(dist_cfg, 'num_workers', 2)
    args.distributed = getattr(dist_cfg, 'distributed', False)
    args.data_parallel = getattr(dist_cfg, 'data_parallel', False)
    
    dataset_kwargs_cache = getattr(dataset, '_kwargs_cache', None)
    
    # Ensure required image and text layers are available
    required_img_layers = ['final'] + list(getattr(cfg.model, 'image_layer_names', []))
    required_txt_layers = ['final'] + list(getattr(cfg.model, 'text_layer_names', []))
    
    # Initialize loss_kwargs if not provided
    if loss_kwargs is None:
        loss_kwargs = {}

    # Trim to required layers when stores are regular dicts
    image_embeddings  = _filter_required_layers(image_embeddings,  required_img_layers)
    caption_embeddings = _filter_required_layers(caption_embeddings, required_txt_layers)

    train_indices = split_dict['train']['indices']
    val_indices   = split_dict['val']['indices']
    train_caption_indices = [dataset.get_idx_to_ptr(i) for i in train_indices]
    val_caption_indices   = [dataset.get_idx_to_ptr(i) for i in val_indices]
    train_neg_indices = [dataset.get_idx_to_candidates_ptr(i) for i in train_indices]
    val_neg_indices   = [dataset.get_idx_to_candidates_ptr(i) for i in val_indices]
    train_neg_dict = {train_indices[i]: train_neg_indices[i] for i in range(len(train_indices))}
    val_neg_dict   = {val_indices[i]:   val_neg_indices[i]   for i in range(len(val_indices))}

    if is_main_process():
        wandb.init(
            project="component-clip-abl" if not train_cfg.test or not "test" in cfg.exp_name else "component-clip-abltest",
            name=f"{cfg.exp_name}-hnb"
        )
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Add barrier after wandb init to ensure all processes wait for it to complete
    if getattr(cfg.dist, 'distributed', False) and dist.is_available() and dist.is_initialized():
        if is_main_process():
            logging.info("Rank 0: wandb initialized, entering barrier")
        dist.barrier()
        logging.info(f"Rank {get_rank()}: All processes synchronized after wandb init")

    use_lazy = _pick_lazy_mode(image_embeddings, caption_embeddings)
    if use_lazy and is_main_process():
        logging.info("Using lazy loading for chunked/tar-based embeddings")

    train_data, val_data = _build_train_val_datasets(
        use_lazy, image_embeddings, caption_embeddings,
        train_neg_dict, val_neg_dict,
        train_indices, val_indices,
        train_caption_indices, val_caption_indices,
        dataset=dataset, args=args,
        dtype=torch.float32
    )


    # Memory monitoring after dataset creation
    memory_stats = log_memory_usage("After Dataset Creation", get_rank())
    log_memory_trend(memory_stats, "After Dataset Creation")

    is_distributed = bool(getattr(cfg.dist, 'distributed', False))
    train_dl, val_dl, train_sampler, val_sampler = _build_dataloaders(args, train_data, val_data, is_distributed)

    # Memory monitoring after dataloader creation
    memory_stats = log_memory_usage("After Dataloader Creation", get_rank())
    log_memory_trend(memory_stats, "After Dataloader Creation")

    
    # Infer layer layouts
    image_layer_names, image_layer_dims = infer_layer_layout(image_embeddings)
    text_layer_names,  text_layer_dims  = infer_layer_layout(caption_embeddings)

    # Sync config across ranks
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        
        image_layer_names, text_layer_names, image_layer_dims, text_layer_dims = \
            sync_model_config_across_ranks(image_layer_names, text_layer_names, image_layer_dims, text_layer_dims, device)

    # Build model using modular head system with config-driven initialization
    head, batch_unpack_fn, loss_fn, evaluate_fn, mode = create_model_from_config(
        cfg=cfg,
        image_layer_dims=image_layer_dims,
        text_layer_dims=text_layer_dims,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        dtype=torch.float32,
        is_ft=False,  # LabCLIP uses pre-extracted features
    )
    
    # Wrap head in feature pipeline (no encoder, just feature processing)
    # Note: CLIPFeaturePipeline doesn't need layer names since it works with pre-extracted features
    model = CLIPFeaturePipeline(
        head=head,
    ).to(device)
    
    gpu_wrapper = MultiGPUWrapper(model, args)
    model = gpu_wrapper.get_model()

    # Memory monitoring after model creation
    memory_stats = log_memory_usage("After Model Creation", get_rank())
    log_memory_trend(memory_stats, "After Model Creation")

    # DDP param sync
    if is_distributed and dist.is_available() and dist.is_initialized():
        logging.info(f"Rank {get_rank()}: Entering sync_ddp_params")
        sync_ddp_params(gpu_wrapper)
        logging.info(f"Rank {get_rank()}: sync_ddp_params completed")

    # Debug: Log before optimizer creation
    logging.info(f"Rank {get_rank()}: About to create optimizer and scheduler")

    # Optimizer/scheduler
    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg, train_dl)

    # Debug: Log before training
    logging.info(f"Rank {get_rank()}: Optimizer and scheduler created, about to start training")

    # Memory monitoring before training starts
    memory_stats = log_memory_usage("Before Training", get_rank())

    # Extract config sections for cleaner parameter access
    train_cfg = cfg.training
    eval_cfg = cfg.evaluation  
    align_cfg = cfg.alignment
    loss_cfg = cfg.loss
    logger_cfg = cfg.logger

    # Initialize sample logger for tracking model learning on fixed samples
    sample_logger = None
    if is_main_process() and getattr(logger_cfg, 'enable_sample_logging', False):
        try:
            from utils.training_sample_logger import TrainingSampleLogger
            num_samples = getattr(logger_cfg, 'sample_logging_num_samples', 10)
            max_components = getattr(logger_cfg, 'sample_logging_max_components', 5)
            logging.info(f"Initializing TrainingSampleLogger with {num_samples} samples, max {max_components} components...")
            sample_logger = TrainingSampleLogger(
                dataset=dataset,
                base_clip_model=clip_model,
                preprocess=preprocess,
                device=device,
                num_samples=num_samples,
                max_components=max_components,
                seed=train_cfg.seed,
                is_ft=False  # LabCLIP uses pre-extracted features
            )
            logging.info(f"✅ Sample logger initialized successfully with {len(sample_logger.samples)} valid samples")
        except Exception as e:
            logging.warning(f"Failed to initialize sample logger (non-critical): {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            sample_logger = None

    # Train - pass config sections directly, training function extracts what it needs
    best_model_dict = train_model_multigpu_merged_batch(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        batch_unpack_fn=batch_unpack_fn,
        evaluate_fn=evaluate_fn,
        cfg=cfg,
        gpu_wrapper=gpu_wrapper,
        train_sampler=train_sampler,
        scheduler=scheduler,
        log_fn=wandb.log,
        loss_kwargs=loss_kwargs,
        sample_logger=sample_logger,
        clip_model=clip_model,
        preprocess=preprocess,
        
        # Pass config sections directly - training function handles parameter extraction
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        align_cfg=align_cfg,
        loss_cfg=loss_cfg,
        logger_cfg=logger_cfg,
    )

    if is_main_process():
        logging.info("Training complete. Evaluating results...")
        logging.info(f"Model alphas (weights): {gpu_wrapper.get_base_model().get_alphas()}")

    # Memory monitoring after training completes
    memory_stats = log_memory_usage("After Training", get_rank())

    return gpu_wrapper.get_base_model(), best_model_dict

def run_ft_clip(dataset, 
                model_clip, 
                preprocess,
                split_dict, 
                device, 
                cfg,
                loss_kwargs=None):
    
    # Extract config sections for cleaner access
    train_cfg = cfg.training
    eval_cfg = cfg.evaluation  
    align_cfg = cfg.alignment
    dataset_cfg = cfg.dataset
    
    # Create minimal args for utility functions that haven't been converted yet
    import types
    args = types.SimpleNamespace()
    args.dataset = dataset_cfg.name
    args.batch_size = train_cfg.batch_size
    args.use_tar_batching = getattr(dataset_cfg, 'use_tar_batching', True)
    args.learning_rate = cfg.optimizer.learning_rate
    args.distributed = getattr(cfg.dist, 'distributed', False)
    args.data_parallel = getattr(cfg.dist, 'data_parallel', False)
    
    is_distributed = bool(getattr(cfg.dist, 'distributed', False))
    
    # Initialize loss_kwargs if not provided
    if loss_kwargs is None:
        loss_kwargs = {}

    if is_main_process():
        wandb.init(
            project="component-clip-abl" if not train_cfg.test or not "test" in cfg.exp_name else "component-clip-abl-test",
            name=f"{cfg.exp_name}-ft"
        )
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Indices & dataloaders
    train_indices = split_dict['train']['indices']
    val_indices   = split_dict['val']['indices']
    train_dl, val_dl, train_sampler, val_sampler = _build_ft_dataloaders(
        args, dataset, train_indices, val_indices, is_distributed
    )

    # Optional: dataloader sanity dump
    if is_main_process() and getattr(train_cfg, 'debug_dataloader', False):
        try:
            logging.info("🔍 Debugging FT DataLoader samples...")
            debug_dump_dataloader_samples(
                train_dl,
                dataset_name=dataset_cfg.name,
                out_dir="ft_debug_samples",
                n_batches=2,
                n_samples_per_batch=3
            )
        except Exception as e:
            logging.warning(f"DataLoader debug sampling failed: {e}")

    # Dtype prep
    image_layer_names = ['final'] + list(getattr(cfg.model, 'image_layer_names', []))
    text_layer_names  = ['final'] + list(getattr(cfg.model, 'text_layer_names', []))
    model_clip, align_dtype = _prep_clip_for_alignment(model_clip, getattr(train_cfg, 'force_float32', False))

    # Probe dims
    image_layer_dims, text_layer_dims = _probe_layer_dims_from_batch(
        train_dl, model_clip, device, image_layer_names, text_layer_names, align_dtype
    )

    # Build FT model wrapper using config-driven initialization
    head, batch_unpack_fn, loss_fn, evaluate_fn, mode = create_model_from_config(
        cfg=cfg,
        image_layer_dims=image_layer_dims,
        text_layer_dims=text_layer_dims,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        dtype=align_dtype,
        is_ft=True,  # Fine-tuning mode
    )
    
    # Create appropriate pipeline based on mode
    if mode == 'text_query_aggregator':
        # TQA mode: use TextQueryAggregatorPipeline which extracts patches
        logging.info("Creating TextQueryAggregatorPipeline for TQA mode")
        model = TextQueryAggregatorPipeline(
            model=model_clip,
            head=head,
            freeze_clip=not (align_cfg.ft_image or align_cfg.ft_text),
            freeze_text_encoder=not align_cfg.ft_text,
            freeze_vision_encoder=not align_cfg.ft_image,
            assume_inputs_on_device=True,
        ).to(device)
    else:
        # Standard mode: use CLIPEndToEndPipeline
        model = CLIPEndToEndPipeline(
            model=model_clip,
            head=head,
            image_layer_names=image_layer_names,
            text_layer_names=text_layer_names,
            ft_image_encoder=align_cfg.ft_image,
            ft_text_encoder=align_cfg.ft_text,
            assume_inputs_on_device=True,
        ).to(device)

    gpu_wrapper = MultiGPUWrapper(model, args)
    model = gpu_wrapper.get_model()

    # DDP param sync (reuse helper if present)
    if is_distributed and dist.is_available() and dist.is_initialized():
        try:
            sync_ddp_params(gpu_wrapper)  # from your HNB/SB refactor
        except NameError:
            pass  # optional

    # Optimizer / scheduler - use the modern cfg-based factory
    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg, train_dl)

    # Extract config sections for cleaner parameter access
    train_cfg = cfg.training
    eval_cfg = cfg.evaluation  
    align_cfg = cfg.alignment
    loss_cfg = cfg.loss
    logger_cfg = cfg.logger

    # Initialize sample logger for FT mode
    sample_logger = None
    if is_main_process() and getattr(logger_cfg, 'enable_sample_logging', False):
        try:
            from utils.training_sample_logger import TrainingSampleLogger
            num_samples = getattr(logger_cfg, 'sample_logging_num_samples', 10)
            max_components = getattr(logger_cfg, 'sample_logging_max_components', 5)
            logging.info(f"Initializing TrainingSampleLogger (FT mode) with {num_samples} samples, max {max_components} components...")
            sample_logger = TrainingSampleLogger(
                dataset=dataset,
                base_clip_model=model_clip,
                preprocess=preprocess,
                device=device,
                num_samples=num_samples,
                max_components=max_components,
                seed=train_cfg.seed,
                is_ft=True  # Fine-tuning mode
            )
            logging.info(f"✅ Sample logger (FT) initialized successfully with {len(sample_logger.samples)} valid samples")
        except Exception as e:
            logging.warning(f"Failed to initialize sample logger (non-critical): {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            sample_logger = None

    # Train - pass config sections directly, training function extracts what it needs
    best_model_dict = train_model_multigpu_merged_batch(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        batch_unpack_fn=batch_unpack_fn,
        evaluate_fn=evaluate_fn,
        cfg=cfg,
        gpu_wrapper=gpu_wrapper,
        train_sampler=train_sampler,
        scheduler=scheduler,
        log_fn=lambda x: wandb.log(x) if is_main_process() and wandb.run else None,
        loss_kwargs=loss_kwargs,
        sample_logger=sample_logger,
        clip_model=model_clip,
        preprocess=preprocess,
        
        # Pass config sections directly - training function handles parameter extraction
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        align_cfg=align_cfg,
        loss_cfg=loss_cfg,
        logger_cfg=logger_cfg,
    )
    return gpu_wrapper.get_base_model(), best_model_dict




@hydra.main(version_base=None, config_path="configs", config_name="hnb_alignment")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # Add some replacements for easier access
    cfg.epochs = cfg.training.epochs
    OmegaConf.set_struct(cfg, True)
    
    # Extract config sections for cleaner access throughout main function
    # Note: These are used by main_with_cfg which is called from main
    
    # Log the CFG for main processes in a structured way
    if is_main_process():
        log_omegaconf_config(cfg)


    # Handle distributed training setup
    spawn_config = handle_distributed_training(cfg)
    if spawn_config:
        # Use multiprocessing.spawn
        mp.spawn(
            spawn_config['spawn_func'],
            args=spawn_config['spawn_args'],
            nprocs=spawn_config['world_size'],
            join=True
        )
        return
    
    # Continue with main training logic
    main_with_cfg(cfg)
    



def handle_distributed_training(cfg):
    """Handle distributed training setup for both torch.distributed.launch and multiprocessing.spawn modes."""
    from utils.dist import is_distributed_launch_mode
    
    if is_distributed_launch_mode():
        # torch.distributed.launch mode - run main directly
        print("Detected torch.distributed.launch mode")
        return None  # Signal to run main directly
    elif cfg.dist.distributed:
        # multiprocessing.spawn mode
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print("Distributed requested but only 1 GPU available. Proceeding with single GPU.")
            cfg.dist.distributed = False
            return None  # Signal to run main directly
        else:
            print("Using multiprocessing.spawn mode")
            # Clear any existing distributed environment variables to avoid conflicts
            for env_var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
                if env_var in os.environ:
                    del os.environ[env_var]
                    print(f"Cleared existing {env_var} environment variable")
            
            # Set master port BEFORE spawning processes to ensure all processes use the same port
            if cfg.dist.master_port is not None:
                os.environ['MASTER_PORT'] = str(cfg.dist.master_port)
                print(f"Using specified master port: {cfg.dist.master_port}")
            elif 'MASTER_PORT' not in os.environ:
                import random
                master_port = random.randint(12000, 65000)
                os.environ['MASTER_PORT'] = str(master_port)
                print(f"Using random master port: {master_port}")
            else:
                print(f"Using master port from environment: {os.environ['MASTER_PORT']}")
            
            # Set master address if not already set
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            
            # Debug: print final environment variables
            print(f"Final MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT_SET')}")
            print(f"Final MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT_SET')}")
            
            # Return spawn config
            return {
                'world_size': world_size,
                'spawn_func': distributed_train_wrapper,
                'spawn_args': (world_size, cfg, main_with_cfg)
            }
    else:
        # Single GPU mode
        return None  # Signal to run main directly


def build_loss_kwargs_from_cfg(cfg, device):
    """
    Build loss_kwargs dictionary from configuration.
    
    DEPRECATED: Use alignment.loss_factory.create_loss_from_config() instead.
    
    This function is kept for backward compatibility but will be removed in future versions.
    
    Args:
        cfg: Configuration object with loss settings
        device: Device for loading projection matrix
        
    Returns:
        dict: loss_kwargs dictionary for loss function
    """
    from alignment.loss_factory import create_loss_from_config
    
    logging.warning(
        "build_loss_kwargs_from_cfg() is deprecated. "
        "Use alignment.loss_factory.create_loss_from_config() instead."
    )
    
    # Use the new factory function
    _, loss_kwargs = create_loss_from_config(cfg)
    
    return loss_kwargs


def main_with_cfg(cfg):
    """Single entry point for both single- and multi-GPU runs (Hydra-friendly).
    Handles torch.distributed.launch and non-launch runs, then executes the full pipeline.
    """
    
    # Extract config sections for cleaner access throughout function
    train_cfg = cfg.training
    align_cfg = cfg.alignment
    dataset_cfg = cfg.dataset
    dist_cfg = cfg.dist
    cache_cfg = cfg.cache
    
    # Create minimal args for compatibility with utility functions (temporary)
        # ---- Distributed setup (supports torch.distributed.launch and non-launch runs) ----
    from utils.dist import is_distributed_launch_mode, get_distributed_info, setup_distributed
    import types
    from omegaconf import OmegaConf
    args = types.SimpleNamespace()
    
    if is_distributed_launch_mode():
        # Running under torch.distributed.launch
        rank, local_rank, world_size = get_distributed_info()
        setup_distributed(rank, world_size)
        # Update cfg with distributed info
        OmegaConf.set_struct(cfg, False)
        cfg.dist.rank = rank
        cfg.dist.local_rank = local_rank
        cfg.device_str = f"cuda:{local_rank}"  # Store as string, not torch.device
        cfg.dist.distributed = True
        OmegaConf.set_struct(cfg, True)
        torch.cuda.set_device(local_rank)
        logging.info(f"torch.distributed.launch mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Basic required fields - using config sections
    args.dataset = dataset_cfg.name
    args.data_path = dataset_cfg.data_path
    args.subset_name = dataset_cfg.subset_name
    args.cache_folder = cache_cfg.cache_folder
    args.recompute_cache = cache_cfg.recompute_cache
    args.val_ratio = dataset_cfg.val_ratio
    args.seed = train_cfg.seed
    args.split_type = dataset_cfg.split_type
    args.num_workers = getattr(dist_cfg, 'num_workers', 2)
    args.data_parallel = getattr(dist_cfg, 'data_parallel', False)
    args.load_checkpoint = getattr(train_cfg, 'load_checkpoint', None)
    
    # Training fields - using config sections
    args.alignment_type = align_cfg.alignment_type
    args.distributed = dist_cfg.distributed
    args.embedding_batch_size = cache_cfg.embedding_batch_size
    args.batch_size = train_cfg.batch_size
    args.use_best_model = train_cfg.use_best_model
    args.load_checkpoint = train_cfg.load_checkpoint
    args.ft_image = align_cfg.ft_image
    args.ft_text = align_cfg.ft_text
    
    if hasattr(dataset_cfg, 'dataset_kwargs'):
        args.dataset_kwargs = OmegaConf.to_container(dataset_cfg.dataset_kwargs, resolve=True)
    else:
        args.dataset_kwargs = {}
    
    # ---- Device selection ----
    if hasattr(cfg, "device_str") and cfg.device_str is not None:
        device = torch.device(cfg.device_str)
    elif hasattr(dist_cfg, "local_rank") and dist_cfg.local_rank is not None and dist_cfg.local_rank >= 0:
        device = torch.device(f"cuda:{dist_cfg.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = getattr(dist_cfg, "rank", 0)
    logging.info(f"Rank {rank}: Using device: {device}")

    if device.type == "cuda":
        current_device = torch.cuda.current_device()
        if current_device != device.index:
            logging.warning(f"Rank {rank}: Current CUDA device ({current_device}) != expected device ({getattr(device, 'index', None)})")

    # ---- Initial memory monitoring ----
    try:
        log_memory_usage("Script Start", force_gc=False)
        logging.info("Initial memory usage logged")
    except Exception as e:
        logging.warning(f"Failed to log initial memory: {e}")

    # ---- Repro / seed ----
    set_seed(train_cfg.seed)

    # ---- Load CLIP ----
    logging.info(f"Using CLIP model: {cfg.model.clip_model}")
    try:
        clip_model, preprocess = load_clip_model(
            model_name=cfg.model.clip_model,
            device=device,
            download_root="/mnt/lustre/work/oh/owl336/.cache/clip",
            force_openclip=True
        )
        logging.info(f"Successfully loaded CLIP model: {cfg.model.clip_model}")
        logging.info(f"Model type: {clip_model.model_type}")
        if hasattr(clip_model, 'model_name'):
            logging.info(f"Full model name: {clip_model.model_name}")

    except Exception as e:
        logging.error(f"Failed to load CLIP model {cfg.model.clip_model}: {e}")
        clip_model, preprocess = clip.load(
            cfg.model.clip_model,
            device=device,
            download_root="/mnt/lustre/work/oh/owl336/.cache/clip",
        )

    # ---- Experiment naming & config snapshot (main process only) ----
    exp_name = f"{train_cfg.name}"
    
    # Make config temporarily mutable to add exp_name
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.exp_name = exp_name  # Store in cfg for later use
    OmegaConf.set_struct(cfg, True)
    
    logging.info(f"Experiment name: {exp_name}")

    if is_main_process():
        model_checkpoint_folder = os.path.join(train_cfg.save_path, f"{exp_name}")
        os.makedirs(model_checkpoint_folder, exist_ok=True)
        with open(os.path.join(model_checkpoint_folder, "config.json"), "w") as f:
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            json.dump(config_dict, f, indent=4)

    # ---- Validate requested intermediate layer names (friendly errors) ----
    if cfg.model.image_layer_names:
        logging.info(f"Requested image layers: {cfg.model.image_layer_names}")
        missing_image_layers = [
            name for name in cfg.model.image_layer_names
            if name not in dict(clip_model.visual.named_modules())
        ]
        if missing_image_layers:
            logging.error(f"Missing image layer names: {missing_image_layers}")
            logging.error("Available image layers (showing some examples):")
            for i, (name, _) in enumerate(clip_model.visual.named_modules()):
                if i < 20:
                    logging.error(f"  - {name}")
                elif i == 20:
                    logging.error("  ... (and more)")
                    break
            raise ValueError(f"Invalid image layer names: {missing_image_layers}")

    if cfg.model.text_layer_names:
        logging.info(f"Requested text layers: {cfg.model.text_layer_names}")
        missing_text_layers = [
            name for name in cfg.model.text_layer_names
            if name not in dict(clip_model.transformer.named_modules())
        ]
        if missing_text_layers:
            logging.error(f"Missing text layer names: {missing_text_layers}")
            logging.error("Available text layers (showing some examples):")
            for i, (name, _) in enumerate(clip_model.transformer.named_modules()):
                if i < 20:
                    logging.error(f"  - {name}")
                elif i == 20:
                    logging.error("  ... (and more)")
                    break
            raise ValueError(f"Invalid text layer names: {missing_text_layers}")

    intermediate_text_layer_names = list(cfg.model.text_layer_names or []) + ["final"]
    intermediate_image_layer_names = list(cfg.model.image_layer_names or []) + ["final"]

    logging.info(f"Using device: {device}")

    # ---- Optional cache invalidation before dataset build ----
    if args.recompute_cache and is_main_process():
        logging.info("Clearing all caches due to --recompute_cache flag")
        from data_loading.laion import LAION400MDataset
        LAION400MDataset.clear_index_cache()
        logging.info("Cleared LAION index caches")

    # ---- Dataset ----
    dataset = build_dataset_from_args(args, preprocess)
    logging.info(f"Using dataset: {args.dataset}")

    # ---- Embedding cache naming ----
    folder = args.cache_folder
    image_embedding_path = get_dataset_cache_name(args, cache_folder=folder, is_image=True, is_dict=True)
    text_embedding_path  = get_dataset_cache_name(args, cache_folder=folder, is_image=False, is_dict=True)
    logging.info(f"If necessary, image embeddings will be saved to: {image_embedding_path}")

    # ---- Split creation ----
    if args.dataset == "LAION400M":
        if is_main_process():
            logging.info("Using tar-based splitting for LAION400M dataset.")
        split_dict = dataset.split_by_tar(val_ratio=args.val_ratio, test_ratio=0.0, seed=args.seed)
    else:
        logging.info(f"Using standard split for dataset: {args.dataset}")
        split_dict = dataset.split_dataset(
            val_ratio=args.val_ratio, test_ratio=0.0, split_type=args.split_type, seed=args.seed
        )

    # ---- Sync after split ----
    if getattr(args, "distributed", False) and dist.is_initialized():
        if is_main_process():
            logging.info(
                f"Split created: train={len(split_dict['train']['indices'])}, val={len(split_dict['val']['indices'])}"
            )
        if not safe_barrier():
            logging.warning("Failed to sync after split creation, continuing anyway")
        else:
            logging.info("All processes synchronized after split creation")

    # ---- LAION optimization: split-scoped vocab (lower RAM) ----
    if args.dataset == "LAION400M":
        if is_main_process():
            logging.info("Preparing split-scoped caption vocabulary for LAION dataset to reduce RAM usage")
            logging.info("This builds caption list only for train/val splits (reduces from full corpus to ~few million strings)")
            logging.info("Dropping compact index cache to ensure consistency...")
            logging.info(f"About to call prepare_caption_vocab_for_splits on main process (rank {get_rank()})")
            # Only main process builds the vocabulary
            dataset.prepare_caption_vocab_for_splits(split_dict, include_splits=("train", "val"))
            logging.info("Main process completed vocabulary preparation")
        else:
            logging.info(f"Worker process (rank {get_rank()}) waiting for vocab preparation")
        
        # CRITICAL: All processes must wait for vocabulary to be built before accessing it
        if args.distributed and dist.is_initialized():
            logging.info(f"Process rank {get_rank()} waiting for vocab barrier")
            if not safe_barrier():
                logging.error("Failed to sync after vocabulary preparation - this may cause shared memory issues")
                raise RuntimeError("Vocabulary preparation synchronization failed")
            else:
                logging.info(f"Process rank {get_rank()} passed vocab barrier")
        
        # After barrier, worker processes can connect to the shared vocabulary
        if not is_main_process():
            if hasattr(dataset, "connect_to_shared_vocab_after_build"):
                logging.info(f"Worker process rank {get_rank()} connecting to shared vocab")
                dataset.connect_to_shared_vocab_after_build(
                    split_dict=split_dict, 
                    include_splits=("train", "val"), 
                    verbose=True, 
                    max_wait_seconds=36000
                )
        
        # Final validation that shared memory is accessible on all processes
        if args.distributed and dist.is_initialized():
            if hasattr(dataset, "_unified_indices") and hasattr(dataset, "_vocab_size"):
                try:
                    logging.info(f"Rank {get_rank()}: Post-sync validation: {dataset._vocab_size} captions accessible")
                    # Test access to unified indices
                    if dataset._unified_indices is not None and len(dataset._unified_indices) > 0:
                        logging.info(f"Rank {get_rank()}: ✅ Shared memory access confirmed - unified indices loaded")
                    else:
                        logging.warning(f"Rank {get_rank()}: ⚠️ Shared memory access failed - unified indices empty or None")
                except Exception as e:
                    logging.warning(f"Rank {get_rank()}: Post-sync shared memory test failed: {e}")
        
        if is_main_process():
            logging.info("Split-scoped vocab preparation complete")
    
    # ---- Save sample visualization for component sampling verification ----
    if is_main_process() and hasattr(dataset, 'save_sample_visualization'):
        try:
            logging.info("Saving sample visualization for component sampling verification...")
            dataset.save_sample_visualization(
                output_dir="./laion_temp",
                num_samples=5,
                random_samples=True,
                seed=args.seed,
                verbose=True
            )
            logging.info("✅ Sample visualization saved to ./laion_temp/")
            logging.info("   Check images and captions to verify component sampling is working correctly")
        except Exception as e:
            logging.warning(f"Sample visualization failed (non-critical): {e}")

    # ---- Optionally clean stale embedding cache files ----
    if args.recompute_cache and is_main_process():
        embedding_dir = os.path.dirname(image_embedding_path)
        image_embedding_basename = os.path.basename(image_embedding_path)
        text_embedding_basename  = os.path.basename(text_embedding_path)
        image_base_pattern = image_embedding_basename.replace("img", "*").replace(".pt", "")
        text_base_pattern  = text_embedding_basename.replace("text", "*").replace(".pt", "")
        clean_up_cache(embedding_dir, image_base_pattern)
        clean_up_cache(embedding_dir, text_base_pattern)

    # ---- Debug sample dump (main only) ----
    try:
        if is_main_process():
            out_dir = "temp"
            debug_dump_samples(dataset, split_dict, preprocess, out_dir=out_dir, n_train=5, n_val=5, seed=args.seed)
            logging.info(f"Dataset debug samples saved to {out_dir}/ (images + captions)")
    except Exception as e:
        if is_main_process():
            logging.warning(f"Debug dump failed: {e}")

    # ---- Dataset pre-evaluation (optional) ----
    prev_results = {}
    after_results = {}

    if hasattr(dataset, "evaluate"):
        if getattr(args, "distributed", False) and dist.is_initialized():
            if not safe_barrier():
                logging.warning("Failed to sync before pre-evaluation, continuing anyway")

        if is_main_process():
            logging.info("Dataset has evaluate method, running evaluation before alignment.")
            with torch.inference_mode():
                results, _ = dataset.evaluate(
                    embedding_model=clip_model,
                    indices=split_dict["val"]["indices"],
                    intermediate_image_layer_names=intermediate_image_layer_names,
                    intermediate_text_layer_names=intermediate_text_layer_names,
                )
            prev_results = flatten_dict(results, parent_key="eval_dataset")
            logging.info(f"Evaluation results: {results}")

        if getattr(args, "distributed", False) and dist.is_initialized():
            if not safe_barrier():
                logging.warning("Failed to sync after pre-evaluation, continuing anyway")

    # ---- Loss kwargs (shared) ----
    loss_kwargs = build_loss_kwargs_from_cfg(cfg, device)

    # ---- Training branches ----
    best_model_dict = None

    if args.alignment_type in ["HNB", "SB", "TCA"]:
        # TCA does not use cached embeddings - it computes features on-the-fly
        if args.alignment_type == "TCA":
           raise NotImplementedError("TCA alignment type is not yet implemented in this script.")
        else:
            # HNB/SB modes: compute and cache embeddings first
            # Warn if using stale cache in DDP
            if args.distributed and not args.recompute_cache:
                if os.path.exists(image_embedding_path) or os.path.exists(image_embedding_path.replace(".pt", "_metadata.json")):
                    logging.warning("Distributed training detected with existing embedding cache.")
                    logging.warning("If you're experiencing low training accuracy, consider using --recompute_cache")

            is_distributed = getattr(args, "distributed", False) and dist.is_initialized()

            # Build per-rank dataset for embedding computation
            embedding_batch_size = args.embedding_batch_size or args.batch_size
            if is_distributed:
                world_size = dist.get_world_size()
                r = dist.get_rank()
                dataset_size = len(dataset)
                indices_per_rank = (dataset_size + world_size - 1) // world_size
                start_idx = r * indices_per_rank
                end_idx = min(start_idx + indices_per_rank, dataset_size)
                rank_indices = list(range(start_idx, end_idx))
                from torch.utils.data import Subset
                rank_dataset = Subset(dataset, rank_indices)
                effective_batch_size = embedding_batch_size
                if is_main_process():
                    logging.info("Distributed embedding computation:")
                    logging.info(f"  - Dataset size: {dataset_size}")
                logging.info(f"  - Rank {r} processing indices: {start_idx} to {end_idx-1}")
                if is_main_process():
                    logging.info(f"  - Per-GPU batch size: {effective_batch_size}")
                    logging.info(f"  - Total effective batch size: {embedding_batch_size * world_size}")
                    logging.info(f"  - World size: {world_size}")
            else:
                logging.info("Single-GPU embedding computation (not distributed)")
                effective_batch_size = embedding_batch_size
                rank_dataset = dataset

            # Dataloader params for embedding streaming
            embedding_num_workers = getattr(args, "num_workers", 2)
            tar_loader_params = get_optimized_dataloader_params("LAION400M", embedding_num_workers)
            embedding_num_workers = tar_loader_params["num_workers"]

            if is_main_process():
                logging.info(f"Embedding computation - num_workers: {embedding_num_workers} for single rank.")

            dataset_loader = DataLoader(
                rank_dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=embedding_num_workers,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=4,
            )

            # Prepare/load existing embeddings (prefer lazy)
            image_embeddings, caption_embeddings = prepare_distributed_embeddings(
                is_distributed=is_distributed,
                args=args,
                image_embedding_path=image_embedding_path,
                text_embedding_path=text_embedding_path,
            )

            clip_model.eval()

            # IMAGE embeddings
            needs_img_embed = (
                image_embeddings is None
                or args.recompute_cache
                or (not is_lazy_embedding_loader(image_embeddings) and any(ln not in image_embeddings for ln in intermediate_image_layer_names))
                or (is_lazy_embedding_loader(image_embeddings) and any(ln not in image_embeddings.layer_names for ln in intermediate_image_layer_names))
            )
            if needs_img_embed:
                logging.info("Image embeddings not found or recompute flag is set. Computing image embeddings with streaming cache.")
                if is_main_process():
                    logging.info(f"Dataset size: {len(dataset)}, Effective batch size: {effective_batch_size}")
                    logging.info(f"Distributed: {is_distributed}, World size: {dist.get_world_size() if is_distributed else 1}")
                    
                image_embeddings = compute_image_embeddings_streaming(
                    image_embeddings=image_embeddings,
                    image_embedding_path=image_embedding_path,
                    args=args,
                    dataset=dataset,                # full dataset; fn handles rank sharding
                    dataset_loader=dataset_loader,  # per-rank (or single) loader
                    clip_model=clip_model,
                    device=device,
                    intermediate_image_layer_names=intermediate_image_layer_names,
                    is_distributed=is_distributed,
                    stream_rows_per_file=5000,
                )

            if is_main_process() and image_embeddings is not None:
                debug_validate_embedding_size(image_embeddings, len(dataset), name="Image Embeddings")
                debug_validate_embedding_order(
                    image_embeddings, dataset, clip_model, device, args,
                    sample_indices=np.random.choice(len(dataset), size=min(10, len(dataset)), replace=False),
                )

            if is_distributed:
                dist.barrier()
                logging.info(f"Rank {dist.get_rank()}: Synchronized after image embedding computation")

            # CAPTION embeddings
            needs_cap_embed = (
                caption_embeddings is None
                or args.recompute_cache
                or (not is_lazy_embedding_loader(caption_embeddings) and any(ln not in caption_embeddings for ln in intermediate_text_layer_names))
                or (is_lazy_embedding_loader(caption_embeddings) and any(ln not in caption_embeddings.layer_names for ln in intermediate_text_layer_names))
            )
            if needs_cap_embed:
                caption_embeddings = compute_caption_embeddings_streaming(
                    caption_embeddings=caption_embeddings,
                    caption_embeddings_path=text_embedding_path,
                    args=args,
                    dataset=dataset,
                    clip_model=clip_model,
                    device=device,
                    intermediate_text_layer_names=intermediate_text_layer_names,
                    is_distributed=is_distributed,
                    stream_rows_per_file=50000,
                )
                if is_distributed:
                    dist.barrier()
                logging.info(f"Rank {rank}: All ranks have loaded full caption embeddings")

            clip_model.train()

            if is_main_process() and caption_embeddings is not None:
                if hasattr(dataset, "get_caption_count"):
                    total_captions = dataset.get_caption_count()
                elif hasattr(dataset, "get_captions_for_embedding"):
                    total_captions = len(dataset.get_captions_for_embedding())
                else:
                    total_captions = 0

                if is_lazy_embedding_loader(caption_embeddings):
                    debug_validate_embedding_size(caption_embeddings, total_captions, name="Caption Embeddings)")
                    debug_check_nan_embeddings(caption_embeddings, name="captions", sample_size=2048)
                    logging.info("Validating caption embedding order and consistency...")
                    debug_validate_caption_embedding_order(
                        dataset=dataset,
                        caption_embeddings=caption_embeddings,
                        clip_model=clip_model,
                        device=device,
                        args=args,
                    )
                    logging.info("Analyzing caption locality and index spans...")
                    debug_caption_locality(dataset=dataset, sample_size=1000)
                    if hasattr(dataset, "inspect_samples"):
                        logging.info("Inspecting random samples for caption quality verification...")
                        dataset.inspect_samples(num_samples=5, random_seed=42, verbose=True)
                    if hasattr(dataset, "inspect_tar_locality"):
                        logging.info("Inspecting tar locality for vocabulary ordering verification...")
                        dataset.inspect_tar_locality(num_tars=3, samples_per_tar=3, random_seed=42, verbose=True)

            if getattr(args, "distributed", False) and dist.is_initialized():
                if not safe_barrier():
                    logging.warning("Failed to sync after debug validation, continuing anyway")
                else:
                    logging.info("All processes synchronized after debug validation")

            if image_embeddings is not None and not is_lazy_embedding_loader(image_embeddings):
                image_embeddings = {k: v.detach().to("cpu") for k, v in image_embeddings.items()}
            if caption_embeddings is not None and not is_lazy_embedding_loader(caption_embeddings):
                caption_embeddings = {k: v.detach().to("cpu") for k, v in caption_embeddings.items()}

            if getattr(args, "distributed", False) and dist.is_initialized():
                logging.info(f"Rank {rank}: About to enter training phase")
                if not safe_barrier():
                    logging.warning("Failed to sync before training, continuing anyway")
                else:
                    logging.info(f"Rank {rank}: All processes synchronized before training")

            # ---- Train alignment (HNB/SB with cached embeddings) ----
            model, best_model_dict = run_labclip(
                dataset, image_embeddings, caption_embeddings, split_dict, device, cfg, clip_model, preprocess, loss_kwargs
            )

        # Common post-training steps for HNB/SB/TCA
        if getattr(args, "distributed", False) and dist.is_initialized():
            dist.barrier()

        # ---- Optionally load best model ----
        if args.use_best_model and is_main_process():
            if best_model_dict is not None:
                logging.info("Loading best model from validation loss.")
                model.load_state_dict(best_model_dict["best_model_state_dict"])
                model.to(device)
            else:
                logging.info("No best model found, using the last checkpoint.")
                if args.load_checkpoint:
                    checkpoint_data = torch.load(args.load_checkpoint, map_location=device)
                    model.load_state_dict(checkpoint_data)
                    model.to(device)
                else:
                    logging.warning("No checkpoint provided, using the model as is.")

        if getattr(args, "distributed", False) and dist.is_initialized():
            if not safe_barrier():
                logging.warning("Failed to sync after model loading, continuing anyway")

        if getattr(args, "distributed", False) and dist.is_initialized():
            if not monitor_nccl_health():
                logging.error("NCCL health check failed before post-evaluation")
                logging.warning("Proceeding with evaluation despite NCCL issues")

        # ---- Post-evaluation ----
        if hasattr(dataset, "evaluate"):
            if is_main_process():
                logging.info("Dataset has evaluate method, running evaluation after alignment.")
                with torch.inference_mode():
                    results, _ = dataset.evaluate(
                        embedding_model=clip_model,
                        aligning_model=model,
                        indices=split_dict["val"]["indices"],
                        intermediate_image_layer_names=intermediate_image_layer_names,
                        intermediate_text_layer_names=intermediate_text_layer_names,
                    )
                after_results = flatten_dict(results, parent_key="eval_dataset")
                logging.info(f"Evaluation results: {results}")

            if getattr(args, "distributed", False) and dist.is_initialized():
                if not safe_barrier():
                    logging.warning("Failed to sync after post-evaluation, continuing anyway")

    else:
        # ---- FT alignment branch ----
        if args.ft_image is False and args.ft_text is False:
            raise ValueError("At least one of ft_image or ft_text must be True for FT alignment.")

        model, best_model_dict = run_ft_clip(dataset, clip_model, preprocess, split_dict, device, cfg, loss_kwargs)

        if getattr(args, "distributed", False) and dist.is_initialized():
            if not safe_barrier():
                logging.warning("Failed to sync after FT training, continuing anyway")

        try:
            if args.use_best_model and is_main_process():
                if best_model_dict is not None:
                    logging.info("Loading best model from validation loss.")
                    model.load_state_dict(best_model_dict["best_model_state_dict"])
                    model.to(device)
                else:
                    logging.info("No best model found, using the last checkpoint.")
                    if args.load_checkpoint:
                        checkpoint_data = torch.load(args.load_checkpoint, map_location=device)
                        model.load_state_dict(checkpoint_data)
                        model.to(device)
                    else:
                        logging.warning("No checkpoint provided, using the model as is.")
        except Exception as e:
            logging.error(f"Failed to load best model: {e}")
            logging.warning("Proceeding with current model weights.")

        if getattr(args, "distributed", False) and dist.is_initialized():
            if not safe_barrier():
                logging.warning("Failed to sync after FT model loading, continuing anyway")
            if not monitor_nccl_health():
                logging.error("NCCL health check failed before FT evaluation")
                logging.warning("Proceeding with FT evaluation despite NCCL issues")

        if hasattr(dataset, "evaluate"):
            if is_main_process():
                logging.info("Dataset has evaluate method, running evaluation after FT alignment.")
                with torch.inference_mode():
                    results, _ = dataset.evaluate(
                        embedding_model=model,
                        indices=split_dict["val"]["indices"],
                        intermediate_image_layer_names=intermediate_image_layer_names,
                        intermediate_text_layer_names=intermediate_text_layer_names,
                    )
                after_results = flatten_dict(results, parent_key="eval_dataset")
                logging.info(f"Evaluation results: {results}")

            if getattr(args, "distributed", False) and dist.is_initialized():
                if not safe_barrier():
                    logging.warning("Failed to sync after FT evaluation, continuing anyway")

    # ---- Log before/after results to wandb (main only) ----
    if prev_results and after_results and is_main_process():
        logging.info("Logging results before and after alignment.")
        logging.info(f"Previous Results: {prev_results}")
        logging.info(f"After Results: {after_results}")
        prev_results.update({"evaluate_step": 0})
        after_results.update({"evaluate_step": 1})
        wandb.log(prev_results)
        wandb.log(after_results)

    # ---- Save checkpoints (main only) ----
    try:
        if args.save_path and is_main_process():
            model_checkpoint_folder = os.path.join(args.save_path, f"{args.exp_name}")
            os.makedirs(model_checkpoint_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_checkpoint_folder, "last_checkpoint.pt"))
            if best_model_dict is not None and best_model_dict.get("best_model_state_dict", None) is not None:
                torch.save(best_model_dict["best_model_state_dict"], os.path.join(model_checkpoint_folder, "best_checkpoint.pt"))
            else:
                logging.warning("No best_model_state_dict found, only saving last_checkpoint.pt.")
            with open(os.path.join(model_checkpoint_folder, "config.json"), "w") as f:
                config_dict = {}
                for k, v in vars(args).items():
                    try:
                        json.dumps(v)
                        config_dict[k] = v
                    except (TypeError, ValueError):
                        config_dict[k] = str(v)
                json.dump(config_dict, f, indent=4)
    except Exception as e:
        if is_main_process():
            logging.error(f"Failed to save checkpoints: {e}")



if __name__ == "__main__":
    # Check if this is being called directly (not through Hydra)
    import sys
    if len(sys.argv) > 1 and not any(arg.startswith('--config-path') or arg.startswith('--config-name') for arg in sys.argv):
        # This might be a direct call, try to handle backward compatibility
        print("Warning: Consider using Hydra configs instead of command line arguments")
        print("Example: python align.py --config-name=hnb_alignment")
        print("Falling back to legacy argument parsing...")
        # You could add fallback argparse here if needed
        exit(1)
    
    # Normal Hydra execution
    main()
