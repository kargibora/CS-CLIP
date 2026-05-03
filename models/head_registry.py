"""
Head registry system for modular head management.
Provides a central registry for registering and instantiating different head types.
"""

from __future__ import annotations
import logging
from typing import Dict, Type, Optional, Any, Callable
from .base_heads import BaseHead, AlignmentHead, BimodalHead


class HeadRegistry:
    """
    Registry for managing different head implementations.
    Allows dynamic registration and instantiation of heads by name.
    """
    
    _alignment_heads: Dict[str, Type[AlignmentHead]] = {}
    _bimodal_heads: Dict[str, Type[BimodalHead]] = {}
    _factories: Dict[str, Callable] = {}
    
    @classmethod
    def register_alignment_head(cls, name: str):
        """Decorator to register an alignment head class."""
        def decorator(head_class: Type[AlignmentHead]):
            cls._alignment_heads[name] = head_class
            logging.debug(f"Registered alignment head: {name}")
            return head_class
        return decorator
    
    @classmethod
    def register_bimodal_head(cls, name: str):
        """Decorator to register a bimodal head class."""
        def decorator(head_class: Type[BimodalHead]):
            cls._bimodal_heads[name] = head_class
            logging.debug(f"Registered bimodal head: {name}")
            return head_class
        return decorator
    
    @classmethod
    def register_factory(cls, name: str, factory_fn: Callable):
        """Register a factory function for creating heads."""
        cls._factories[name] = factory_fn
        logging.debug(f"Registered factory: {name}")
    
    @classmethod
    def create_alignment_head(cls, name: str, **kwargs) -> AlignmentHead:
        """Create an alignment head by name."""
        if name not in cls._alignment_heads:
            raise ValueError(
                f"Unknown alignment head: {name}. "
                f"Available: {list(cls._alignment_heads.keys())}"
            )
        return cls._alignment_heads[name](**kwargs)
    
    @classmethod
    def create_bimodal_head(cls, name: str, **kwargs) -> BimodalHead:
        """Create a bimodal head by name."""
        if name not in cls._bimodal_heads:
            raise ValueError(
                f"Unknown bimodal head: {name}. "
                f"Available: {list(cls._bimodal_heads.keys())}"
            )
        return cls._bimodal_heads[name](**kwargs)
    
    @classmethod
    def create_head(cls, head_type: str, name: str, **kwargs) -> BaseHead:
        """Create a head by type and name."""
        if head_type == 'alignment':
            return cls.create_alignment_head(name, **kwargs)
        elif head_type == 'bimodal':
            return cls.create_bimodal_head(name, **kwargs)
        else:
            raise ValueError(f"Unknown head type: {head_type}. Available: alignment, bimodal")
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseHead:
        """
        Create a head from configuration dictionary.
        
        Config format:
            {
                'type': 'alignment' or 'bimodal',
                'name': 'multi_layer_aggregator',
                'params': { ... }
            }
        """
        if 'type' not in config or 'name' not in config:
            raise ValueError("Config must contain 'type' and 'name' fields")
        
        head_type = config['type']
        name = config['name']
        params = config.get('params', {})
        
        factory_key = f"{head_type}_{name}"
        if factory_key in cls._factories:
            return cls._factories[factory_key](**params)
        
        return cls.create_head(head_type, name, **params)
    
    @classmethod
    def list_heads(cls, head_type: Optional[str] = None) -> Dict[str, list]:
        """List all registered heads, optionally filtered by type."""
        all_heads = {
            'alignment': list(cls._alignment_heads.keys()),
            'bimodal': list(cls._bimodal_heads.keys()),
        }
        
        if head_type:
            if head_type not in all_heads:
                raise ValueError(f"Unknown head type: {head_type}")
            return {head_type: all_heads[head_type]}
        
        return all_heads
