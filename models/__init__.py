"""
Models package for CLIP alignment and fine-tuning.
"""

from .alignment_heads import CLIPFeatureAlignment
from .aggregators import MultiLayerCLIPAggregator
from .base_heads import BaseHead, AlignmentHead, BimodalHead
from .head_registry import HeadRegistry
from .clip_ft import FlexibleCLIPMultiLayerAlignment, CLIPMultiLayerFTAlignment
from .pipelines import (
    BaseCLIPPipeline,
    CLIPEndToEndPipeline,
)

__all__ = [
    # Base classes
    'BaseHead',
    'AlignmentHead',
    'BimodalHead',
    
    # Registry
    'HeadRegistry',
    
    # Alignment heads
    'CLIPFeatureAlignment',
    'MultiLayerCLIPAggregator',
    
    # Bimodal heads
    'FlexibleCLIPMultiLayerAlignment',
    
    # Full models
    'CLIPMultiLayerFTAlignment',
    
    # Pipelines
    'BaseCLIPPipeline',
    'CLIPEndToEndPipeline',
    
    # Factory functions
    'get_model_by_name',
    'get_model_by_config',
    'create_head',
]


def get_model_by_name(name: str, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        name: Model name ('clip_ft', 'flexible_alignment')
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    """
    models = {
        'clip_ft': CLIPMultiLayerFTAlignment,
        'flexible_alignment': FlexibleCLIPMultiLayerAlignment,
        'aggregator': MultiLayerCLIPAggregator,
        'alignment_head': CLIPFeatureAlignment,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model name: {name}. Available: {list(models.keys())}")
    
    return models[name](**kwargs)


def get_model_by_config(config: dict):
    """
    Factory function to instantiate model from config dict.
    
    Config format:
        {
            'type': 'clip_ft',  # or 'flexible_alignment', 'cross_modal_mlp'
            'params': {
                # model-specific parameters
            }
        }
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Instantiated model
    """
    if 'type' not in config:
        raise ValueError("Config must contain 'type' field")
    
    model_type = config['type']
    params = config.get('params', {})
    
    return get_model_by_name(model_type, **params)


def create_head(head_type: str, name: str, **kwargs):
    """
    Factory function to create a head from the registry.
    
    Args:
        head_type: Type of head ('alignment', 'bimodal')
        name: Name of the registered head
        **kwargs: Head-specific parameters
    
    Returns:
        Instantiated head
    """
    return HeadRegistry.create_head(head_type, name, **kwargs)
