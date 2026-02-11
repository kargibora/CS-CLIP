"""Utilities for device management and batch handling."""

from typing import Any, Dict, Union
import torch


def move_to_device(x: Union[torch.Tensor, Dict[str, torch.Tensor]], device: torch.device):
    """
    Move tensor or dict of tensors to device.
    
    Args:
        x: Tensor or dict of tensors to move
        device: Target device
        
    Returns:
        Tensor or dict on target device
    """
    if isinstance(x, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device)
    return x


def ensure_correct_device(device: torch.device, rank: int) -> torch.device:
    """
    Ensure device matches the current rank's CUDA device.
    
    Args:
        device: Requested device
        rank: Current process rank
        
    Returns:
        Corrected device
    """
    if torch.cuda.is_available() and device.type == 'cuda':
        current_cuda_device = torch.cuda.current_device()
        if device.index != current_cuda_device:
            return torch.device(f'cuda:{current_cuda_device}')
    return device


def get_base_model(model: Any) -> Any:
    """
    Unwrap DDP-wrapped model to get the base model.
    
    Args:
        model: Possibly DDP-wrapped model
        
    Returns:
        Base model
    """
    return model.module if hasattr(model, 'module') else model


def is_labclip_model(model: Any) -> bool:
    """
    Check if model is a LabCLIP variant.
    
    Args:
        model: Model to check
        
    Returns:
        True if LabCLIP model
    """
    from .constants import ModelType
    
    base_model = get_base_model(model)
    model_class = base_model.__class__.__name__
    
    is_labclip = (
        model_class == ModelType.LABCLIP_FLEXIBLE or
        (model_class == ModelType.CLIP_MULTILAYER_FT and 
         hasattr(base_model, 'use_labclip') and 
         getattr(base_model, 'use_labclip', False))
    )
    
    return is_labclip


def is_ft_mode(model: Any) -> bool:
    """
    Check if model is in fine-tuning mode.
    
    Args:
        model: Model to check
        
    Returns:
        True if FT mode
    """
    from .constants import ModelType
    
    base_model = get_base_model(model)
    
    if hasattr(base_model, 'use_labclip'):
        return getattr(base_model, 'use_labclip', False)
    elif hasattr(base_model, '__class__') and ModelType.CLIP_MULTILAYER_FT in str(base_model.__class__):
        return True
    
    return False
