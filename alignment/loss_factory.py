"""
Loss Function Factory for CLIP Alignment Training

Provides a centralized registry for all loss functions with name-based selection.
This allows easy switching between different loss types via configuration.

Supports ComponentLossConfig for clean parameter management.
"""

import logging
import os
import torch.distributed as dist
from typing import Dict, Callable, Any, Optional
from alignment.losses import (
    multi_caption_contrastive_loss,
    ComponentLossConfig,
)


def is_main_process() -> bool:
    """Check if this is the main process (rank 0 or single process)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# Global registry mapping loss names to functions
LOSS_REGISTRY: Dict[str, Callable] = {    
    # Multi-caption contrastive loss (NEW)
    "multi_caption": multi_caption_contrastive_loss,
    "multi_caption_contrastive": multi_caption_contrastive_loss,  # Alias
}


def get_loss_function(loss_name: str) -> Callable:
    """
    Get a loss function by name from the registry.
    
    Args:
        loss_name: Name of the loss function (case-insensitive)
        
    Returns:
        Loss function callable
        
    Raises:
        ValueError: If loss_name is not found in registry
        
    Example:
        >>> loss_fn = get_loss_function("multi_caption")
        >>> loss_dict = loss_fn(image_embeddings, pos_text_embeddings, ...)
    """
    # Normalize name (lowercase, replace hyphens/underscores)
    normalized_name = loss_name.lower().replace('-', '_')
    
    if normalized_name not in LOSS_REGISTRY:
        available = ', '.join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown loss function: '{loss_name}'. "
            f"Available losses: {available}"
        )
    
    return LOSS_REGISTRY[normalized_name]


def register_loss(name: str, loss_fn: Callable) -> None:
    """
    Register a custom loss function.
    
    Args:
        name: Name to register the loss under
        loss_fn: Loss function callable
        
    Example:
        >>> def my_custom_loss(**kwargs):
        ...     return {'loss': ..., 'accuracy': ...}
        >>> register_loss("custom", my_custom_loss)
    """
    normalized_name = name.lower().replace('-', '_')
    if normalized_name in LOSS_REGISTRY:
        logging.warning(f"Overwriting existing loss function: {normalized_name}")
    LOSS_REGISTRY[normalized_name] = loss_fn
    logging.info(f"Registered loss function: {normalized_name}")


def list_available_losses() -> list:
    """
    Get list of all available loss function names.
    
    Returns:
        List of registered loss names
    """
    return sorted(LOSS_REGISTRY.keys())


def get_loss_info(loss_name: str) -> Dict[str, Any]:
    """
    Get information about a specific loss function.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Dictionary with loss function metadata
    """
    loss_fn = get_loss_function(loss_name)
    
    return {
        'name': loss_name,
        'function': loss_fn.__name__,
        'docstring': loss_fn.__doc__,
        'module': loss_fn.__module__,
    }


def create_loss_from_config(cfg) -> tuple[Callable, Dict[str, Any]]:
    """
    Create loss function and kwargs from configuration.
    
    This is the main entry point for creating losses in training scripts.
    Supports both raw kwargs and ComponentLossConfig for component-based losses.
    
    Args:
        cfg: Configuration object with loss settings
        
    Returns:
        Tuple of (loss_function, loss_kwargs)
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("configs/loss/multi_caption.yaml")
        >>> loss_fn, loss_kwargs = create_loss_from_config(cfg)
        
    Config Structure (Option 1 - Direct kwargs):
        loss:
          loss_type: "multi_caption"
          contrastive_mode: "with_components_negatives"
          lambda_full: 1.0           # Weight for full caption loss
          lambda_components: 1.0     # Weight for component caption loss
          lambda_rank: 0.0           # Weight for ranking loss
          tau_sim: 0.1               # Ranking margin
          use_coverage_weights: true
          ranking_weighting_strategy: "coverage_ratio"
          ranking_alpha: 1.0
          
    Config Structure (Option 2 - Using config object):
        loss:
          loss_type: "multi_caption"
          contrastive_mode: "with_components_negatives"
          config:
            lambda_full: 1.0
            lambda_components: 0.5
            lambda_rank: 0.1
            tau_sim: 0.15
            
    Config Structure (Option 3 - Backward compatible with alpha):
        loss:
          loss_type: "multi_caption"
          contrastive_mode: "with_components"
          alpha: 0.6  # Automatically converts to lambda_full=0.6, lambda_components=0.4
          lambda_rank: 0.1
    """
    # Helper for logging only on main process
    def log_info(msg):
        if is_main_process():
            logging.info(msg)
    
    def log_warning(msg):
        if is_main_process():
            logging.warning(msg)
    
    # Get loss configuration
    loss_cfg = getattr(cfg, 'loss', None)
    if loss_cfg is None:
        log_warning("No loss configuration found, using default CLIP loss")
        return get_loss_function("clip"), {}
    
    # Determine loss type
    loss_type = getattr(loss_cfg, 'loss_type', None)
    
    # If no explicit loss_type, infer from training mode
    if loss_type is None:
        training_mode = getattr(cfg, 'mode', 'ft')
        if training_mode == 'labclip':
            loss_type = 'labclip'
        elif training_mode == 'tca':
            loss_type = 'tca'
        else:
            # Check if multi-caption mode is enabled
            dataset_kwargs = getattr(cfg.dataset, 'dataset_kwargs', {})
            num_component_captions = dataset_kwargs.get('num_component_captions', 0)
            if num_component_captions > 0:
                loss_type = 'multi_caption'
                log_info(f"Detected num_component_captions={num_component_captions}, using multi_caption loss")
            else:
                loss_type = 'clip'
    
    # Get loss function
    try:
        loss_fn = get_loss_function(loss_type)
        log_info(f"✓ Using loss function: {loss_type} ({loss_fn.__name__})")
    except ValueError as e:
        logging.error(f"Failed to get loss function: {e}")
        log_warning("Falling back to default CLIP loss")
        loss_fn = get_loss_function("clip")
        loss_type = 'clip'
    
    # Build loss kwargs from config
    # Strategy: Pass all config attributes as kwargs, excluding special fields
    loss_kwargs = {}
    
    # Fields to exclude (these are meta-fields, not loss parameters)
    exclude_fields = {'loss_type', 'projection_matrix_path', 'loss_kwargs', 'config'}
    
    # Option 1: Check if there's a 'config' sub-object (ComponentLossConfig style)
    if hasattr(loss_cfg, 'config') and loss_cfg.config is not None:
        log_info("  - Using ComponentLossConfig-style configuration")
        # Create ComponentLossConfig from the config sub-object
        config_dict = {}
        if hasattr(loss_cfg.config, '__dict__'):
            for key, value in loss_cfg.config.items():
                config_dict[key] = value
        else:
            for key in loss_cfg.config:
                config_dict[key] = loss_cfg.config[key]
        
        # Create config object and convert to dict
        try:
            component_config = ComponentLossConfig(**config_dict)
            loss_kwargs.update(component_config.to_dict())
            log_info(f"  - Created ComponentLossConfig: {component_config}")
        except Exception as e:
            log_warning(f"  - Failed to create ComponentLossConfig: {e}")
            # Fallback: use raw dict
            loss_kwargs.update(config_dict)
    
    # Option 2: Extract all top-level attributes as kwargs (standard approach)
    if hasattr(loss_cfg, '__dict__'):
        # OmegaConf object
        for key, value in loss_cfg.items():
            if key not in exclude_fields and key not in loss_kwargs:
                loss_kwargs[key] = value
    elif hasattr(loss_cfg, '_content'):
        # Handle DictConfig
        for key in loss_cfg:
            if key not in exclude_fields and key not in loss_kwargs:
                loss_kwargs[key] = loss_cfg[key]
    
    # Special handling: merge loss_kwargs if it exists (for backward compatibility)
    if hasattr(loss_cfg, 'loss_kwargs') and loss_cfg.loss_kwargs is not None:
        if isinstance(loss_cfg.loss_kwargs, dict):
            loss_kwargs.update(loss_cfg.loss_kwargs)
        else:
            # OmegaConf DictConfig
            loss_kwargs.update(dict(loss_cfg.loss_kwargs))
    
    # Backward compatibility: Handle alpha parameter
    # If alpha is specified but lambda_full/lambda_components are not, convert it
    if 'alpha' in loss_kwargs and 'lambda_full' not in loss_kwargs:
        alpha_value = loss_kwargs['alpha']
        log_info(f"  - Converting alpha={alpha_value} to lambda parameters")
        loss_kwargs['lambda_full'] = alpha_value
        loss_kwargs['lambda_components'] = 1.0 - alpha_value
        # Keep alpha for backward compatibility, but the loss function will prioritize lambdas
    
    # Validate lambda parameters for component losses
    if loss_type in ['multi_caption', 'multi_caption_contrastive']:
        # Ensure we have the new lambda parameters or can fall back to alpha
        has_lambdas = 'lambda_full' in loss_kwargs or 'lambda_components' in loss_kwargs
        has_alpha = 'alpha' in loss_kwargs
        
        if not has_lambdas and not has_alpha:
            log_warning("  - No lambda_full/lambda_components or alpha found, using defaults (1.0, 1.0)")
            loss_kwargs['lambda_full'] = 1.0
            loss_kwargs['lambda_components'] = 1.0
        
        # Set default component_loss_type if not specified
        if 'component_loss_type' not in loss_kwargs:
            loss_kwargs['component_loss_type'] = 'negclip'  # Default
            log_info("  - component_loss_type not specified, using default: 'negclip'")
        
        # Log the final lambda values
        if 'lambda_full' in loss_kwargs:
            log_info(f"  - lambda_full: {loss_kwargs['lambda_full']}")
        if 'lambda_components' in loss_kwargs:
            log_info(f"  - lambda_components: {loss_kwargs['lambda_components']}")
        if 'lambda_paraphrase' in loss_kwargs:
            log_info(f"  - lambda_paraphrase: {loss_kwargs['lambda_paraphrase']}")
        if 'lambda_rank' in loss_kwargs:
            log_info(f"  - lambda_rank: {loss_kwargs['lambda_rank']}")
        if 'component_loss_type' in loss_kwargs:
            log_info(f"  - component_loss_type: {loss_kwargs['component_loss_type']}")
    
    # Special handling: projection matrix for comprehensive_binding loss
    if loss_type in ['comprehensive_binding', 'binding']:
        projection_matrix_path = getattr(loss_cfg, 'projection_matrix_path', None)
        if projection_matrix_path:
            import torch
            try:
                loss_kwargs['A'] = torch.load(projection_matrix_path)
                log_info(f"  - Loaded projection matrix from: {projection_matrix_path}")
            except Exception as e:
                log_warning(f"  - Failed to load projection matrix: {e}")
                loss_kwargs['A'] = None
        elif 'A' not in loss_kwargs:
            loss_kwargs['A'] = None
    
    # Log important parameters for visibility (only on main process)
    if loss_kwargs and is_main_process():
        log_info("  - Loss parameters:")
        for key, value in sorted(loss_kwargs.items()):
            # Skip already logged parameters
            if key in ['lambda_full', 'lambda_components', 'lambda_paraphrase', 'lambda_rank', 'component_loss_type']:
                continue
            # Skip logging large objects
            if key == 'A' and value is not None:
                log_info(f"    • {key}: <projection_matrix>")
            elif isinstance(value, (int, float, str, bool)):
                log_info(f"    • {key}: {value}")
            elif value is None:
                log_info(f"    • {key}: None")
            else:
                log_info(f"    • {key}: {type(value).__name__}")
    
    return loss_fn, loss_kwargs


def validate_loss_compatibility(loss_type: str, batch_unpack_fn_name: str) -> bool:
    """
    Validate that loss type is compatible with batch unpacking function.
    
    Args:
        loss_type: Name of loss function
        batch_unpack_fn_name: Name of batch unpacking function
        
    Returns:
        True if compatible, False otherwise
    """
    # Define compatibility matrix
    compatibility = {
        'clip': ['unpack_ft_multilayer', 'unpack_neg_multilayer', 'unpack_ft_tqa'],
        'multi_caption': ['unpack_ft_multilayer', 'unpack_ft_tqa'],
        'labclip': ['unpack_labclip_ft_multilayer', 'unpack_labclip_multilayer'],
        'tca': ['unpack_tca'],
    }
    
    normalized_loss = loss_type.lower().replace('-', '_')
    
    # Get compatible unpack functions
    compatible_funcs = []
    for key, funcs in compatibility.items():
        if normalized_loss.startswith(key):
            compatible_funcs = funcs
            break
    
    if not compatible_funcs:
        logging.warning(f"Unknown loss type for compatibility check: {loss_type}")
        return True  # Allow unknown combinations
    
    # Check if batch_unpack_fn is compatible
    is_compatible = any(func in batch_unpack_fn_name for func in compatible_funcs)
    
    if not is_compatible:
        logging.warning(
            f"Potential incompatibility: loss '{loss_type}' typically requires "
            f"batch unpacking: {compatible_funcs}, but using: {batch_unpack_fn_name}"
        )
    
    return is_compatible
