import logging
import math
import os
import types
import torch
import torch.distributed as dist

from data_loading import (
    get_dataset_embedding_class,
)
from data_loading.laion import create_tar_grouped_dataloader
from utils.align import (
    extract_intermediate_features,
)

from utils.dist import (
    create_distributed_dataloader,
    get_world_size,
    is_main_process,

)
from utils.labclip_helpers import get_optimized_dataloader_params
from utils.optimizer_factory import (
    make_optimizer_and_scheduler as make_modular_optimizer_and_scheduler,
)


def patch_clip_layernorm_for_fp16(model):
    """
    Patch OpenAI CLIP's LayerNorm to properly handle fp16 without dtype mismatches.
    """
    def patched_layernorm_forward(self, x):
        if x.dtype == torch.float16:
            weight = self.weight.half() if self.weight.dtype != torch.float16 else self.weight
            bias = self.bias.half() if self.bias.dtype != torch.float16 else self.bias
            return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        else:
            return torch.nn.functional.layer_norm(x.type(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps).type(x.dtype)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.forward = types.MethodType(patched_layernorm_forward, module)
    return model


def _get_optimized_dataloader_params(dataset_name, base_num_workers=2):
    # Reuse your earlier implementation
    return get_optimized_dataloader_params(dataset_name, base_num_workers)

def _build_ft_dataloaders(args, dataset, train_indices, val_indices, is_distributed):
    """Build FT dataloaders (tar-grouped for LAION, standard otherwise)."""
    neg_dataset_class = get_dataset_embedding_class(args.dataset)
    train_data = neg_dataset_class(dataset, train_indices)
    val_data   = neg_dataset_class(dataset, val_indices)

    base_num_workers = getattr(args, 'num_workers', 2)
    params = _get_optimized_dataloader_params(args.dataset, base_num_workers)
    num_workers = params['num_workers']

    if args.dataset == 'LAION400M' and getattr(args, 'use_tar_batching', True):
        if is_main_process():
            logging.info("Using tar-grouped DataLoader for LAION400M FT training")
            logging.info("  - Batches stay within tar files; ranks get disjoint tar subsets")
        train_dl, train_sampler = create_tar_grouped_dataloader(
            train_data, indices=None, batch_size=args.batch_size,
            distributed=is_distributed, num_workers=num_workers,
            shuffle_within_tar=True, shuffle_tars=True, pin_memory=True,
            prefetch_factor=params['prefetch_factor']
        )
        val_dl, val_sampler = create_tar_grouped_dataloader(
            val_data, indices=None, batch_size=args.batch_size,
            distributed=is_distributed, num_workers=min(2, num_workers),
            shuffle_within_tar=False, shuffle_tars=False, pin_memory=True,
            prefetch_factor=2
        )
    else:
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
    return train_dl, val_dl, train_sampler, val_sampler

def _prep_clip_for_alignment(model_clip, force_float32: bool):
    """Cast CLIP to desired dtype and patch LayerNorm for fp16 if needed."""
    if force_float32:
        logging.info("Using float32 due to --force_float32")
        return model_clip.float(), torch.float32
    logging.info("Using float16 for faster training. Use --force_float32 if you encounter NaNs.")
    model_clip = patch_clip_layernorm_for_fp16(model_clip)
    return model_clip.half(), torch.float16

def _probe_layer_dims_from_batch(train_dataloader, model_clip, device, image_layer_names, text_layer_names, dtype):
    """Take one batch to infer layer dims for FT alignment heads."""
    sample_batch = next(iter(train_dataloader))
    if isinstance(sample_batch, dict):
        images = sample_batch["images"]
        tokens = sample_batch["pos_tokens"]
    else:
        images, tokens, *_ = sample_batch

    images = images.to(device)
    tokens = tokens.to(device)
    
    # Handle multi-caption mode: tokens might be [B, 1+N, seq_len]
    # For dimension probing, we only need one caption per image
    if tokens.ndim == 3:
        # Shape: [B, 1+N, seq_len] -> take first caption [B, seq_len]
        tokens = tokens[:, 0, :]
        logging.info(f"Multi-caption mode detected during probing, using first caption. Token shape: {tokens.shape}")

    with torch.inference_mode():
        img_feats  = extract_intermediate_features(images, model_clip, device, image_layer_names, is_image=True,  dtype=dtype)
        text_feats = extract_intermediate_features(tokens, model_clip, device, text_layer_names,  is_image=False, dtype=dtype)

    img_dims  = [img_feats[ln].shape[-1]  for ln in image_layer_names]
    text_dims = [text_feats[ln].shape[-1] for ln in text_layer_names]
    return img_dims, text_dims

def _scale_lr_for_ddp(cfg):
    """Return LR scaled per cfg.optimizer.lr_scaling policy."""
    lr = cfg.optimizer.learning_rate
    if getattr(cfg.dist, 'distributed', False) and dist.is_available() and dist.is_initialized():
        world = get_world_size()
        policy = getattr(cfg.optimizer, 'lr_scaling', 'none')
        if policy == "linear":
            lr *= world
        elif policy == "sqrt":
            lr *= math.sqrt(world)
        if is_main_process():
            logging.info(f"FT training - LR scaling: {policy}, world={world}, final LR={lr}")
    return lr

def _make_ft_optimizer_and_scheduler(model, cfg, train_dataloader, lr):
    """Use modular factory for optimizer and scheduler creation with specified learning rate."""
    # Create a temporary args object from cfg for compatibility with modular factory
    import types
    
    args = types.SimpleNamespace()
    # Map cfg fields to args fields that the factory expects
    args.learning_rate = lr  # Use the scaled lr
    args.optimizer_name = getattr(cfg.optimizer, 'name', 'adamw')
    args.weight_decay = getattr(cfg.optimizer, 'weight_decay', 0.01)
    args.betas = getattr(cfg.optimizer, 'betas', [0.9, 0.999])
    args.eps = getattr(cfg.optimizer, 'eps', 1e-8)
    args.scheduler_name = getattr(cfg.optimizer, 'scheduler', 'cosine')
    args.warmup_steps = getattr(cfg.optimizer, 'warmup_steps', 1000)
    args.max_steps = len(train_dataloader) * cfg.training.epochs
    args.min_lr = getattr(cfg.optimizer, 'min_lr', 0.0)
    args.epochs = cfg.training.epochs
    
    # Use the modular factory
    return make_modular_optimizer_and_scheduler(model, args, train_dataloader)
    