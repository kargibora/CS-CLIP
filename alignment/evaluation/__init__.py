"""
Evaluation module for CLIP and LabCLIP models.

This module provides evaluation functions for:
- Standard CLIP models with embeddings
- Multi-layer CLIP models
- LabCLIP models with score-based evaluation
- Fine-tuning (FT) mode evaluation

Main API:
    - evaluate(): Standard CLIP evaluation
    - evaluate_labclip(): LabCLIP-specific evaluation
    - evaluate_ft(): Fine-tuning evaluation shorthand
"""

from .core import evaluate, evaluate_ft
from .constants import MetricKeys, ModelType
from .utils import is_labclip_model, is_ft_mode

__all__ = [
    # Main evaluation functions
    'evaluate',
    'evaluate_ft',
    
    # Utility functions
    'is_labclip_model',
    'is_ft_mode',
    
    # Constants
    'MetricKeys',
    'ModelType',
]
