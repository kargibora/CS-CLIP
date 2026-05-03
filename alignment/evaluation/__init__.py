"""Evaluation module for CLIP FT models.

Main API:
    - evaluate(): Standard CLIP evaluation
    - evaluate_ft(): Fine-tuning evaluation shorthand
"""

from .core import evaluate, evaluate_ft
from .constants import MetricKeys, ModelType
from .utils import is_ft_mode

__all__ = [
    'evaluate',
    'evaluate_ft',
    'is_ft_mode',
    'MetricKeys',
    'ModelType',
]
