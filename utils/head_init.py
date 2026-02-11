"""
Smart model initialization strategies for LabCLIP.

This module provides utilities for config-driven head initialization,
including automatic unpacking function selection, loss function selection,
and parameter extraction from Hydra configs.
"""

import logging
from typing import Dict, Any, Tuple, Callable, Optional
from omegaconf import DictConfig, OmegaConf
import torch

from models import HeadRegistry
from alignment.loss_factory import get_loss_function, create_loss_from_config
from alignment.learning_alignment import (
    unpack_ft_multilayer,
    unpack_neg_multilayer,
    unpack_ft_tqa,
)
from alignment.evaluation import evaluate, evaluate_ft


class HeadInitializer:
    """Smart head initialization from Hydra configs."""
    
    # Mapping of head modes to unpacking functions and DEFAULT loss functions
    # NOTE: Actual loss function can be overridden via cfg.loss.loss_type
    MODE_CONFIG = {
        'scoring': {
            'default_loss': 'labclip',  # Default, can be overridden
            'evaluate_fn': evaluate,
            'evaluate_fn_ft': evaluate_ft,
            'default_head': 'cross_modal_mlp',
            'registry_method': 'create_scoring_head',
        },
        'bimodal': {
            'unpack_fn': unpack_neg_multilayer,
            'unpack_fn_ft': unpack_ft_multilayer,
            'default_loss': 'comprehensive_binding',  # Default, can be overridden
            'evaluate_fn': evaluate,
            'evaluate_fn_ft': evaluate_ft,
            'default_head': 'flexible_multi_layer',
            'registry_method': 'create_bimodal_head',
        },
        'alignment': {
            'unpack_fn': unpack_neg_multilayer,
            'unpack_fn_ft': unpack_ft_multilayer,
            'default_loss': 'clip',  # Default, can be overridden
            'evaluate_fn': evaluate,
            'evaluate_fn_ft': evaluate_ft,
            'default_head': 'linear',
            'registry_method': 'create_alignment_head',
        },
        'tca': {
            'default_loss': 'tca',  # Default, can be overridden
            'evaluate_fn': evaluate,
            'evaluate_fn_ft': evaluate_ft,
            'default_head': 'token_conditioned',
            'registry_method': 'create_alignment_head',
        },
        'text_query_aggregator': {
            'unpack_fn': unpack_neg_multilayer,
            'unpack_fn_ft': unpack_ft_tqa,  # Uses encode_image(img, text=...) for cross-attention
            'default_loss': 'multi_caption',  # TQA uses standard contrastive loss
            'evaluate_fn': evaluate,
            'evaluate_fn_ft': evaluate_ft,
            'default_head': 'text_query_aggregator',
            'registry_method': 'create_bimodal_head',
        },
    }
    
    @classmethod
    def get_head_mode(cls, cfg: DictConfig) -> str:
        """
        Determine head mode from config.
        
        Priority:
        1. cfg.model.head.mode (explicit)
        2. cfg.model.use_cross_modal_head (backward compatibility)
        3. Default to 'bimodal'
        
        Args:
            cfg: Hydra config
            
        Returns:
            str: Head mode ('scoring', 'bimodal', or 'alignment')
        """
        # Explicit mode configuration
        if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'mode'):
            mode = cfg.model.head.mode
            if mode not in cls.MODE_CONFIG:
                logging.warning(f"Unknown head mode '{mode}', falling back to 'bimodal'")
                return 'bimodal'
            return mode
        
        # Backward compatibility with use_cross_modal_head
        if getattr(cfg.model, 'use_cross_modal_head', False):
            return 'scoring'
        
        # Default to bimodal
        return 'bimodal'
    
    @classmethod
    def get_head_type(cls, cfg: DictConfig, mode: str) -> str:
        """
        Get head type name from config.
        
        Args:
            cfg: Hydra config
            mode: Head mode
            
        Returns:
            str: Head type name for registry
        """
        # Explicit head type in config
        if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'type'):
            return cfg.model.head.type
        
        # Default head type for mode
        return cls.MODE_CONFIG[mode]['default_head']
    
    @classmethod
    def extract_head_kwargs(cls, cfg: DictConfig, mode: str, **base_kwargs) -> Dict[str, Any]:
        """
        Extract head-specific kwargs from config.
        
        Args:
            cfg: Hydra config
            mode: Head mode
            **base_kwargs: Base kwargs (image_layer_dims, text_layer_dims, etc.)
            
        Returns:
            dict: Kwargs for head initialization
        """
        kwargs = dict(base_kwargs)
        
        # Common parameters for all heads
        kwargs['image_layer_names'] = kwargs.get('image_layer_names', [])
        kwargs['text_layer_names'] = kwargs.get('text_layer_names', [])
        
        # Aggregator config (for multi-layer heads)
        if mode in ['scoring', 'bimodal']:
            kwargs['aggregator_cfg'] = {
                'learnable_alphas': getattr(cfg.model, 'learnable_alphas', False),
                'alphas': getattr(cfg.model, 'alphas', None),
                'init_alpha': getattr(cfg.model, 'init_alpha', 0.5),
                'dtype': kwargs.get('dtype', torch.float32),
            }
        
        # Alignment config (for heads with alignment layers)
        if mode in ['scoring', 'bimodal', 'alignment']:
            kwargs['align_cfg'] = {
                'mlp_layers': getattr(cfg.model, 'mlp_layers', 1),
                'out_dim': getattr(cfg.model, 'embedding_dim', 512),
                'dtype': kwargs.get('dtype', torch.float32),
            }
        
        # Mode-specific parameters
        if mode == 'scoring':
            # LabCLIP scoring head parameters
            # Try new head.* config first, then fall back to labclip_* for backward compat
            if hasattr(cfg.model, 'head'):
                kwargs['mlp_hidden_dims'] = getattr(cfg.model.head, 'mlp_hidden_dims', [])
                kwargs['dropout'] = getattr(cfg.model.head, 'dropout', 0.0)
                kwargs['use_alignment'] = getattr(cfg.model.head, 'use_alignment', False)
            else:
                kwargs['mlp_hidden_dims'] = getattr(cfg.model, 'labclip_hidden_dims', [])
                kwargs['dropout'] = getattr(cfg.model, 'labclip_dropout', 0.0)
                kwargs['use_alignment'] = getattr(cfg.model, 'labclip_use_alignment', False)
            kwargs['embed_dim'] = getattr(cfg.model, 'embedding_dim', 512)
            
        elif mode == 'bimodal':
            # Bimodal alignment head parameters
            kwargs['align_image'] = getattr(cfg.alignment, 'align_image', True)
            kwargs['align_text'] = getattr(cfg.alignment, 'align_text', True)
            
            # Temperature parameters (if in config)
            # For HNB (cached features), default to 1.0 (neutral, doesn't scale logits)
            # For FT (end-to-end), default to 0.07 (standard CLIP temperature)
            alignment_type = getattr(cfg.alignment, 'alignment_type', 'HNB')
            default_temp = 1.0 if alignment_type == 'HNB' else 0.07
            
            if hasattr(cfg.model, 'head'):
                kwargs['init_temperature'] = getattr(cfg.model.head, 'init_temperature', default_temp)
                kwargs['freeze_temperature'] = getattr(cfg.model.head, 'freeze_temperature', False)
            else:
                kwargs['init_temperature'] = getattr(cfg.model, 'init_temperature', default_temp)
                kwargs['freeze_temperature'] = getattr(cfg.model, 'freeze_temperature', False)
        
        elif mode == 'alignment':
            # TCA/Token-conditioned alignment head parameters
            # Extract native transformer dimensions for patch/token projections
            image_layer_dims = kwargs.get('image_layer_dims', {})
            text_layer_dims = kwargs.get('text_layer_dims', {})
            
            logging.info(f"[extract_head_kwargs] Alignment mode - image_layer_dims: {image_layer_dims}")
            logging.info(f"[extract_head_kwargs] Alignment mode - text_layer_dims: {text_layer_dims}")
            
            # Get the last layer dimension (before projection to embed_dim)
            # For ViT, this is typically from 'visual_ln_post' or similar
            # For text transformer, from 'ln_final' or similar
            if image_layer_dims:
                # Get any layer dimension (they should all be the same for native transformer)
                kwargs['vision_dim'] = next(iter(image_layer_dims.values()))
                logging.info(f"[extract_head_kwargs] Set vision_dim: {kwargs['vision_dim']}")
            
            if text_layer_dims:
                kwargs['text_dim'] = next(iter(text_layer_dims.values()))
                logging.info(f"[extract_head_kwargs] Set text_dim: {kwargs['text_dim']}")
            
            # Embed dim (output space)
            kwargs['embed_dim'] = getattr(cfg.model, 'embedding_dim', 512)
            
            # Temperature parameters
            alignment_type = getattr(cfg.alignment, 'alignment_type', 'TCA')
            default_temp = 0.07  # TCA uses standard CLIP temperature
            
            if hasattr(cfg.model, 'head'):
                kwargs['initial_temperature'] = getattr(cfg.model.head, 'initial_temperature', default_temp)
                kwargs['learnable_temperature'] = getattr(cfg.model.head, 'learnable_temperature', False)  # Frozen by default
            else:
                kwargs['initial_temperature'] = default_temp
                kwargs['learnable_temperature'] = False  # Frozen by default
            
            # Use alignment config from cfg.alignment (unified with rest of pipeline)
            # align_text/align_image control whether to use global alignment matrices A and B
            kwargs['use_text_alignment'] = getattr(cfg.alignment, 'align_text', False)
            kwargs['use_image_alignment'] = getattr(cfg.alignment, 'align_image', False)
            
            logging.info(f"[extract_head_kwargs] Alignment - align_text: {kwargs['use_text_alignment']}, align_image: {kwargs['use_image_alignment']}")
        
        elif mode == 'tca':
            # TCA (Token-Conditioned Alignment) head parameters
            # Extract native transformer dimensions for patch/token projections
            image_layer_dims = kwargs.get('image_layer_dims', {})
            text_layer_dims = kwargs.get('text_layer_dims', {})
            
            logging.info(f"[extract_head_kwargs] TCA mode - image_layer_dims: {image_layer_dims}")
            logging.info(f"[extract_head_kwargs] TCA mode - text_layer_dims: {text_layer_dims}")
            
            # Get the native transformer dimensions (before projection to embed_dim)
            if image_layer_dims:
                kwargs['vision_dim'] = next(iter(image_layer_dims.values()))
                logging.info(f"[extract_head_kwargs] Set vision_dim: {kwargs['vision_dim']}")
            
            if text_layer_dims:
                kwargs['text_dim'] = next(iter(text_layer_dims.values()))
                logging.info(f"[extract_head_kwargs] Set text_dim: {kwargs['text_dim']}")
            
            # Embed dim (output space)
            kwargs['embed_dim'] = getattr(cfg.model, 'embedding_dim', 512)
            
            # Temperature parameters
            default_temp = 0.07  # TCA uses standard CLIP temperature
            
            if hasattr(cfg.model, 'head'):
                kwargs['initial_temperature'] = getattr(cfg.model.head, 'initial_temperature', default_temp)
                kwargs['learnable_temperature'] = getattr(cfg.model.head, 'learnable_temperature', False)  # Frozen by default
            else:
                kwargs['initial_temperature'] = default_temp
                kwargs['learnable_temperature'] = False  # Frozen by default
            
            # Use alignment config from cfg.alignment (unified with rest of pipeline)
            # align_text/align_image control whether to use global alignment matrices A and B
            kwargs['use_text_alignment'] = getattr(cfg.alignment, 'align_text', True)
            kwargs['use_image_alignment'] = getattr(cfg.alignment, 'align_image', True)
            
            logging.info(f"[extract_head_kwargs] TCA alignment - align_text: {kwargs['use_text_alignment']}, align_image: {kwargs['use_image_alignment']}")
        
        elif mode == 'text_query_aggregator':
            # Text-Query Aggregator (TQA) head parameters
            # TQA uses cross-attention: v_cls' = v_cls + Attn(q=text, k=patches, v=patches)
            image_layer_dims = kwargs.get('image_layer_dims', {})
            text_layer_dims = kwargs.get('text_layer_dims', {})
            
            logging.info(f"[extract_head_kwargs] TQA mode - image_layer_dims: {image_layer_dims}")
            logging.info(f"[extract_head_kwargs] TQA mode - text_layer_dims: {text_layer_dims}")
            
            # Get the native transformer dimensions (before projection to embed_dim)
            # Handle both dict (layer_name -> dim) and list ([dim1, dim2, ...]) formats
            if image_layer_dims:
                if isinstance(image_layer_dims, dict):
                    kwargs['vision_dim'] = next(iter(image_layer_dims.values()))
                elif isinstance(image_layer_dims, (list, tuple)):
                    kwargs['vision_dim'] = image_layer_dims[0] if image_layer_dims else 512
                else:
                    kwargs['vision_dim'] = int(image_layer_dims)
                logging.info(f"[extract_head_kwargs] TQA set vision_dim: {kwargs['vision_dim']}")
            
            if text_layer_dims:
                if isinstance(text_layer_dims, dict):
                    kwargs['text_dim'] = next(iter(text_layer_dims.values()))
                elif isinstance(text_layer_dims, (list, tuple)):
                    kwargs['text_dim'] = text_layer_dims[0] if text_layer_dims else 512
                else:
                    kwargs['text_dim'] = int(text_layer_dims)
                logging.info(f"[extract_head_kwargs] TQA set text_dim: {kwargs['text_dim']}")
            
            # Embed dim (common dimension for attention)
            kwargs['embed_dim'] = getattr(cfg.model, 'embedding_dim', 512)
            
            # TQA-specific parameters (can be overridden via head.params)
            kwargs['num_heads'] = 8  # Default number of attention heads
            kwargs['dropout'] = 0.0  # Default dropout
            kwargs['use_residual'] = True  # v_cls' = v_cls + attn_output
            kwargs['learnable_scale'] = True  # Learnable residual scale
            kwargs['initial_scale'] = 0.1  # Initial scale for residual
            kwargs['use_projection'] = True  # Use projection heads
            kwargs['normalize_output'] = True  # L2 normalize output
            
            # Temperature parameters
            default_temp = 0.07  # Standard CLIP temperature
            
            if hasattr(cfg.model, 'head'):
                kwargs['initial_temperature'] = getattr(cfg.model.head, 'initial_temperature', default_temp)
                kwargs['learnable_temperature'] = getattr(cfg.model.head, 'learnable_temperature', False)
            else:
                kwargs['initial_temperature'] = default_temp
                kwargs['learnable_temperature'] = False  # Frozen by default
            
            logging.info(f"[extract_head_kwargs] TQA - embed_dim: {kwargs['embed_dim']}, num_heads: {kwargs['num_heads']}")
        
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
        Create head from Hydra config with smart defaults.
        
        Args:
            cfg: Hydra config
            image_layer_dims: Dict mapping layer names to dimensions
            text_layer_dims: Dict mapping layer names to dimensions
            image_layer_names: List of image layer names
            text_layer_names: List of text layer names
            dtype: Data type for head
            
        Returns:
            Tuple[head, mode, head_type]: Created head, mode, and head type
        """
        # Determine head mode
        mode = cls.get_head_mode(cfg)
        logging.info(f"Head mode: {mode}")
        
        # Get head type
        head_type = cls.get_head_type(cfg, mode)
        logging.info(f"Creating head type: {head_type}")
        
        # Extract head kwargs
        base_kwargs = {
            'image_layer_dims': image_layer_dims,
            'text_layer_dims': text_layer_dims,
            'image_layer_names': image_layer_names,
            'text_layer_names': text_layer_names,
            'dtype': dtype,  # Used internally for nested configs
        }
        head_kwargs = cls.extract_head_kwargs(cfg, mode, **base_kwargs)
        
        # Remove dtype from top-level kwargs (only used in nested configs like aggregator_cfg)
        # Most heads don't accept dtype as a top-level parameter
        head_kwargs.pop('dtype', None)
        
        # Get registry method for this mode
        registry_method_name = cls.MODE_CONFIG[mode]['registry_method']
        registry_method = getattr(HeadRegistry, registry_method_name)
        
        # Create head
        logging.info(f"Initializing {head_type} with mode={mode}")
        logging.debug(f"Head kwargs: {head_kwargs}")
        
        try:
            head = registry_method(head_type, **head_kwargs)
        except TypeError as e:
            logging.error(f"Failed to create head {head_type} with kwargs: {head_kwargs}")
            logging.error(f"Error: {e}")
            raise
        
        return head, mode, head_type
    
    @classmethod
    def get_training_functions(
        cls,
        mode: str,
        is_ft: bool = False,
        cfg: Optional[DictConfig] = None
    ) -> Tuple[Callable, Callable, Callable]:
        """
        Get unpacking function, loss function, and evaluation function for mode.
        
        Args:
            mode: Head mode ('scoring', 'bimodal', 'alignment', 'tca')
            is_ft: Whether this is fine-tuning (vs pre-extracted features)
            cfg: Optional configuration to override loss function from config
            
        Returns:
            Tuple[unpack_fn, loss_fn, evaluate_fn]
        """
        config = cls.MODE_CONFIG[mode]
        
        # Select unpacking function based on FT or not
        unpack_fn = config['unpack_fn_ft'] if is_ft else config['unpack_fn']
        
        # Select loss function
        # Priority: 1) cfg.loss.loss_type  2) Default for mode
        if cfg is not None:
            try:
                loss_fn, _ = create_loss_from_config(cfg)
                logging.info(f"Using loss from config: {loss_fn.__name__}")
            except Exception as e:
                logging.warning(f"Failed to create loss from config: {e}")
                logging.warning(f"Falling back to default loss for mode '{mode}'")
                loss_fn = get_loss_function(config['default_loss'])
        else:
            # No config provided, use default for mode
            loss_fn = get_loss_function(config['default_loss'])
            logging.info(f"Using default loss for mode '{mode}': {loss_fn.__name__}")
        
        # Evaluation function
        evaluate_fn = config['evaluate_fn_ft'] if is_ft else config['evaluate_fn']
        
        return unpack_fn, loss_fn, evaluate_fn
    
    @classmethod
    def log_head_info(cls, head, mode: str, head_type: str, is_ft: bool = False, cfg: Optional[DictConfig] = None):
        """Log information about the created head."""
        logging.info("=" * 60)
        logging.info("HEAD CONFIGURATION")
        logging.info("=" * 60)
        logging.info(f"Mode: {mode}")
        logging.info(f"Type: {head_type}")
        logging.info(f"Training: {'Fine-tuning' if is_ft else 'Pre-extracted features'}")
        logging.info(f"Head class: {head.__class__.__name__}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in head.parameters())
        trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        # Log mode-specific info
        unpack_fn, loss_fn, eval_fn = cls.get_training_functions(mode, is_ft, cfg)
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
    is_ft: bool = False,
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
        is_ft: Whether this is fine-tuning mode
        
    Returns:
        Tuple[head, unpack_fn, loss_fn, eval_fn, mode]: 
            - head: Created head module
            - unpack_fn: Batch unpacking function
            - loss_fn: Loss function
            - eval_fn: Evaluation function
            - mode: Head mode string
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
    
    # Get training functions (with config for loss selection)
    unpack_fn, loss_fn, eval_fn = HeadInitializer.get_training_functions(mode, is_ft, cfg)
    
    # Log head info
    HeadInitializer.log_head_info(head, mode, head_type, is_ft, cfg)
    
    return head, unpack_fn, loss_fn, eval_fn, mode
