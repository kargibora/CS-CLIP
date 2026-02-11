"""
Base classes for modular heads in CLIP models.
Defines common interfaces for alignment heads, scoring heads, and cross-modal heads.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for all heads.
    Defines the common interface that all heads must implement.
    """
    
    def __init__(self):
        super().__init__()
        self._device = None
    
    @property
    def device(self):
        """Cached device property to avoid repeated parameter iteration."""
        if self._device is None:
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device('cpu')
        return self._device
    
    def to(self, device):
        """Move module to device."""
        super().to(device)
        self._device = device
        return self
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        pass


class AlignmentHead(BaseHead):
    """
    Base class for alignment heads that transform feature representations.
    Used for single-modality feature alignment (e.g., image or text separately).
    
    Input: Features (tensor, list, or dict)
    Output: Aligned embeddings (tensor)
    """
    
    @abstractmethod
    def forward(
        self,
        features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Align features to embedding space.
        
        Args:
            features: Input features (single tensor, list of layers, or dict)
        
        Returns:
            Aligned embeddings (batch, embed_dim)
        """
        pass


class ScoringHead(BaseHead):
    """
    Base class for scoring heads that compute similarity/relevance scores.
    Used for cross-modal scoring (e.g., image-text matching scores).
    
    Input: Image features + Text features
    Output: Scores (tensor)
    """
    
    @abstractmethod
    def forward(
        self,
        image_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        text_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute cross-modal scores.
        
        Args:
            image_features: Image features
            text_features: Text features
        
        Returns:
            Scores (batch_size,) for matching pairs
        """
        pass


class BimodalHead(BaseHead):
    """
    Base class for heads that process both modalities.
    Can be used for either alignment or scoring, depending on implementation.
    
    Input: Image features + Text features (optional)
    Output: Either (image_embed, text_embed) or scores
    """
    
    @abstractmethod
    def encode_image(
        self,
        image_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Encode image features to embeddings."""
        pass
    
    @abstractmethod
    def encode_text(
        self,
        text_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Encode text features to embeddings."""
        pass
    
    def forward(
        self,
        image_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        text_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass through both modalities.
        
        Args:
            image_features: Image features (optional)
            text_features: Text features (optional)
        
        Returns:
            Embeddings or scores
        """
        img = self.encode_image(image_features) if image_features is not None else None
        txt = self.encode_text(text_features) if text_features is not None else None
        
        if img is not None and txt is not None:
            return img, txt
        return img if txt is None else txt


class FeatureProcessor:
    """
    Utility class for processing features in different formats.
    Handles conversion between dict, list, and tensor formats.
    """
    
    @staticmethod
    def extract_final_feature(
        features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        layer_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Extract the 'final' feature from various input formats.
        Preference: dict['final'] -> dict[last named layer] -> last entry -> tensor itself.
        
        Args:
            features: Input features in any format
            layer_names: Optional layer names for dict lookup
        
        Returns:
            Final feature tensor
        """
        if isinstance(features, dict):
            if "final" in features:
                return features["final"]
            if layer_names and layer_names[-1] in features:
                return features[layer_names[-1]]
            last_key = next(reversed(features))
            return features[last_key]
        elif isinstance(features, list):
            return features[-1]
        else:
            return features  # already a tensor
    
    @staticmethod
    def dict_to_ordered_list(
        features_dict: Dict[str, torch.Tensor],
        layer_names: List[str]
    ) -> List[torch.Tensor]:
        """
        Convert feature dictionary to ordered list based on layer names.
        
        Args:
            features_dict: Dictionary of features
            layer_names: Ordered list of layer names
        
        Returns:
            Ordered list of feature tensors
        """
        missing = [k for k in layer_names if k not in features_dict]
        if missing:
            raise KeyError(
                f"Missing required feature(s): {missing}. "
                f"Available keys: {list(features_dict.keys())}"
            )
        return [features_dict[k] for k in layer_names]
    
    @staticmethod
    def normalize(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        L2-normalize tensor with numerical stability.
        
        Args:
            tensor: Input tensor
            eps: Epsilon for numerical stability
        
        Returns:
            Normalized tensor
        """
        return F.normalize(tensor, dim=-1, eps=eps)
