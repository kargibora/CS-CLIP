"""
Base classes for modular heads in CLIP models.
Defines common interfaces for alignment heads and bimodal heads.
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
