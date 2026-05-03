"""
Checkpoint utilities for CLIP fine-tuning.

Handles saving and loading checkpoints with proper state dict key management.
Deals with DDP wrapping (module.*) and pipeline wrapping (model.*).
"""

import logging
import os
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


def get_base_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DDP/DataParallel wrapper to get base model.
    
    Args:
        model: Potentially wrapped model
        
    Returns:
        Unwrapped base model
    """
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    return model


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean state dict by removing DDP/pipeline wrapper prefixes.
    
    Handles patterns:
    - module.* (DDP wrapper)
    - model.module.* (DDP wrapped CLIPEndToEndPipeline)
    - _orig_mod.* (torch.compile)
    
    The result has keys starting with 'model.' for CLIP backbone params
    and 'head.' for alignment head params.
    
    Args:
        state_dict: Raw state dict from checkpoint
        
    Returns:
        Cleaned state dict
    """
    cleaned = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Remove DDP wrapper prefix
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        
        # Remove torch.compile wrapper
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[len('_orig_mod.'):]
        
        # Handle nested model.module (DDP-wrapped pipeline)
        if new_key.startswith('model.module.'):
            new_key = 'model.' + new_key[len('model.module.'):]
        
        cleaned[new_key] = value
    
    return cleaned


def save_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metric: Optional[float] = None,
    config: Optional[Dict] = None,
) -> None:
    """
    Save checkpoint with clean state dict.
    
    Automatically unwraps DDP and saves the state dict with consistent keys.
    
    Args:
        model: Model to save (may be DDP-wrapped)
        path: Path to save checkpoint
        optimizer: Optional optimizer to save
        scheduler: Optional scheduler to save
        epoch: Optional epoch number
        step: Optional step number
        metric: Optional metric value (e.g., best val loss)
        config: Optional config dict to save
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Get base model (unwrap DDP)
    base_model = get_base_model(model)
    
    # Get state dict
    state_dict = base_model.state_dict()
    
    # Build checkpoint dict
    checkpoint = {
        'model_state_dict': state_dict,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if step is not None:
        checkpoint['step'] = step
    
    if metric is not None:
        checkpoint['metric'] = metric
    
    if config is not None:
        checkpoint['config'] = config
    
    # Save
    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load checkpoint with automatic state dict cleaning.
    
    Handles various wrapper patterns and loads into the model.
    
    Args:
        model: Model to load into (may be DDP-wrapped)
        path: Path to checkpoint
        device: Device to load to
        strict: If True, require exact key match
        
    Returns:
        Dictionary with any extra checkpoint data (epoch, step, metric, config)
    """
    # Load checkpoint
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(path, map_location=device)
    
    # Get state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'best_model_state_dict' in checkpoint:
        state_dict = checkpoint['best_model_state_dict']
    else:
        # Assume the whole checkpoint is the state dict
        state_dict = checkpoint
        checkpoint = {'model_state_dict': state_dict}
    
    # Clean state dict
    state_dict = clean_state_dict(state_dict)
    
    # Get base model (unwrap DDP)
    base_model = get_base_model(model)
    
    # Load state dict
    missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        logging.warning(f"Missing keys in checkpoint: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                logging.warning(f"  - {key}")
    
    if unexpected_keys:
        logging.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                logging.warning(f"  - {key}")
    
    logging.info(f"Loaded checkpoint from {path}")
    
    # Return extra info
    return {
        'epoch': checkpoint.get('epoch'),
        'step': checkpoint.get('step'),
        'metric': checkpoint.get('metric'),
        'config': checkpoint.get('config'),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
    }


def save_best_model(model: nn.Module, path: str) -> None:
    """
    Save just the best model state dict (minimal checkpoint).
    
    Args:
        model: Model to save
        path: Path to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_model = get_base_model(model)
    torch.save(base_model.state_dict(), path)
    logging.info(f"Saved best model to {path}")


def load_best_model(model: nn.Module, path: str, device: torch.device = None) -> None:
    """
    Load a minimal state dict checkpoint.
    
    Args:
        model: Model to load into
        path: Path to checkpoint
        device: Device to load to
    """
    if device is None:
        device = next(model.parameters()).device
    
    state_dict = torch.load(path, map_location=device)
    state_dict = clean_state_dict(state_dict)
    
    base_model = get_base_model(model)
    base_model.load_state_dict(state_dict, strict=False)
    logging.info(f"Loaded model from {path}")
