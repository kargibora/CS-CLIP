"""
Evaluation functions for CLIP models.

This module is now a thin wrapper around the refactored evaluation package.
All implementation has been moved to alignment/evaluation/ for better organization.

For direct imports, use:
    from alignment.evaluation import evaluate, evaluate_labclip, evaluate_ft
"""

# Re-export all main functions from the new package for backwards compatibility
from alignment.evaluation import (
    evaluate,
    evaluate_labclip,
    evaluate_ft,
    MetricKeys,
    ModelType,
    is_labclip_model,
    is_ft_mode,
)

__all__ = [
    'evaluate',
    'evaluate_labclip',
    'evaluate_ft',
    'MetricKeys',
    'ModelType',
    'is_labclip_model',
    'is_ft_mode',
]
