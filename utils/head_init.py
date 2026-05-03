"""Head initialization for the FT CLIP path."""

import logging
from typing import Dict, Any, Tuple, Callable, Optional
from omegaconf import DictConfig, OmegaConf
import torch

from models.clip_ft import FlexibleCLIPMultiLayerAlignment
from alignment.loss_factory import create_loss_from_config
from alignment.learning_alignment import unpack_ft_multilayer
from alignment.evaluation import evaluate_ft


class HeadInitializer:
    """Head initialization for the FT-only bimodal CLIP path."""
    
    @classmethod
    def extract_head_kwargs(cls, cfg: DictConfig, **base_kwargs) -> Dict[str, Any]:
        """
        Extract head kwargs from config for bimodal mode.
        
        Args:
            cfg: Hydra config
            **base_kwargs: Base kwargs (image_layer_dims, text_layer_dims, etc.)
            
        Returns:
            dict: Kwargs for head initialization
        """
        kwargs = dict(base_kwargs)
        
        # Common parameters
        kwargs['image_layer_names'] = kwargs.get('image_layer_names', [])
        kwargs['text_layer_names'] = kwargs.get('text_layer_names', [])
        
        # Aggregator config (for multi-layer heads)
        kwargs['aggregator_cfg'] = {
            'learnable_alphas': getattr(cfg.model, 'learnable_alphas', False),
            'alphas': getattr(cfg.model, 'alphas', None),
            'init_alpha': getattr(cfg.model, 'init_alpha', 0.5),
            'dtype': kwargs.get('dtype', torch.float32),
        }
        
        # Alignment config
        kwargs['align_cfg'] = {
            'mlp_layers': getattr(cfg.model, 'mlp_layers', 1),
            'out_dim': getattr(cfg.model, 'embedding_dim', 512),
            'dtype': kwargs.get('dtype', torch.float32),
        }
        
        # Bimodal alignment head parameters
        kwargs['align_image'] = getattr(cfg.alignment, 'align_image', True)
        kwargs['align_text'] = getattr(cfg.alignment, 'align_text', True)
        
        # Temperature parameters - FT default is 0.07 (standard CLIP temperature)
        default_temp = 0.07
        
        if hasattr(cfg.model, 'head'):
            kwargs['init_temperature'] = getattr(cfg.model.head, 'init_temperature', default_temp)
            kwargs['freeze_temperature'] = getattr(cfg.model.head, 'freeze_temperature', False)
        else:
            kwargs['init_temperature'] = getattr(cfg.model, 'init_temperature', default_temp)
            kwargs['freeze_temperature'] = getattr(cfg.model, 'freeze_temperature', False)
        
        # Allow config to override any parameter via head.params
        if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'params'):
            head_params = OmegaConf.to_container(cfg.model.head.params, resolve=True)
            kwargs.update(head_params)
        
        return kwargs
    
    @classmethod
    def create_head_from_config(
        cls,
        cfg: DictConfig,
        image_layer_dims: Dict[str, int],
        text_layer_dims: Dict[str, int],
        image_layer_names: list,
        text_layer_names: list,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Create bimodal head from Hydra config.
        
        Args:
            cfg: Hydra config
            image_layer_dims: Dict mapping layer names to dimensions
            text_layer_dims: Dict mapping layer names to dimensions
            image_layer_names: List of image layer names
            text_layer_names: List of text layer names
            dtype: Data type for head
            
        Returns:
            Tuple[head, mode, head_type]: Created head, mode='bimodal', and head type
        """
        mode = 'bimodal'
        
        head_type = (
            cfg.model.head.type
            if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'type')
            else 'flexible_multi_layer'
        )
        logging.info(f"Creating head type: {head_type}")
        if head_type != 'flexible_multi_layer':
            raise ValueError(
                f"Unsupported FT head type: {head_type}. "
                "The cleaned FT path only supports 'flexible_multi_layer'."
            )
        
        # Extract head kwargs
        base_kwargs = {
            'image_layer_dims': image_layer_dims,
            'text_layer_dims': text_layer_dims,
            'image_layer_names': image_layer_names,
            'text_layer_names': text_layer_names,
            'dtype': dtype,
        }
        head_kwargs = cls.extract_head_kwargs(cfg, **base_kwargs)
        
        # Remove dtype from top-level kwargs (only used in nested configs)
        head_kwargs.pop('dtype', None)
        
        logging.info(f"Initializing {head_type} with mode={mode}")
        logging.debug(f"Head kwargs: {head_kwargs}")
        try:
            head = FlexibleCLIPMultiLayerAlignment(**head_kwargs)
        except TypeError as e:
            logging.error(f"Failed to create head {head_type} with kwargs: {head_kwargs}")
            logging.error(f"Error: {e}")
            raise
        
        return head, mode, head_type
    
    @classmethod
    def get_training_functions(
        cls,
        cfg: Optional[DictConfig] = None
    ) -> Tuple[Callable, Callable, Callable]:
        """
        Get unpacking function, loss function, and evaluation function.
        
        Args:
            cfg: Optional configuration to get loss function from config
            
        Returns:
            Tuple[unpack_fn, loss_fn, evaluate_fn]
        """
        # Always use FT unpacking function
        unpack_fn = unpack_ft_multilayer
        
        # Get loss function from config
        if cfg is not None:
            try:
                loss_fn, _ = create_loss_from_config(cfg)
                logging.info(f"Using loss from config: {loss_fn.__name__}")
            except Exception as e:
                logging.warning(f"Failed to create loss from config: {e}")
                # Import here to avoid circular imports
                from alignment.loss_factory import get_loss_function
                loss_fn = get_loss_function('multi_caption')
        else:
            from alignment.loss_factory import get_loss_function
            loss_fn = get_loss_function('multi_caption')
            logging.info(f"Using default loss: {loss_fn.__name__}")
        
        # Always use FT evaluation function
        evaluate_fn = evaluate_ft
        
        return unpack_fn, loss_fn, evaluate_fn
    
    @classmethod
    def log_head_info(cls, head, head_type: str, cfg: Optional[DictConfig] = None):
        """Log information about the created head."""
        logging.info("=" * 60)
        logging.info("HEAD CONFIGURATION")
        logging.info("=" * 60)
        logging.info(f"Mode: bimodal (FT)")
        logging.info(f"Type: {head_type}")
        logging.info(f"Head class: {head.__class__.__name__}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in head.parameters())
        trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        # Log functions
        unpack_fn, loss_fn, eval_fn = cls.get_training_functions(cfg)
        logging.info(f"Unpacking function: {unpack_fn.__name__}")
        logging.info(f"Loss function: {loss_fn.__name__}")
        logging.info(f"Evaluation function: {eval_fn.__name__}")
        logging.info("=" * 60)


def create_model_from_config(
    cfg: DictConfig,
    image_layer_dims: Dict[str, int],
    text_layer_dims: Dict[str, int],
    image_layer_names: list,
    text_layer_names: list,
    dtype: torch.dtype = torch.float32,
    is_ft: bool = True,  # Always True for this simplified version
):
    """
    Create complete model (head + functions) from config.
    
    This is the main entry point for config-driven model creation.
    
    Args:
        cfg: Hydra config
        image_layer_dims: Dict mapping layer names to dimensions
        text_layer_dims: Dict mapping layer names to dimensions  
        image_layer_names: List of image layer names
        text_layer_names: List of text layer names
        dtype: Data type for model
        is_ft: Whether this is fine-tuning mode (always True)
        
    Returns:
        Tuple[head, unpack_fn, loss_fn, eval_fn, mode]: 
            - head: Created head module
            - unpack_fn: Batch unpacking function
            - loss_fn: Loss function
            - eval_fn: Evaluation function
            - mode: Head mode string ('bimodal')
    """
    # Create head from config
    head, mode, head_type = HeadInitializer.create_head_from_config(
        cfg=cfg,
        image_layer_dims=image_layer_dims,
        text_layer_dims=text_layer_dims,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        dtype=dtype,
    )
    
    # Get training functions
    unpack_fn, loss_fn, eval_fn = HeadInitializer.get_training_functions(cfg)
    
    # Log head info
    HeadInitializer.log_head_info(head, head_type, cfg)
    
    return head, unpack_fn, loss_fn, eval_fn, mode
