"""
Modular pipeline base classes for CLIP models.
Provides common interfaces for attaching heads to CLIP models.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

from .base_heads import BaseHead, BimodalHead
from .head_registry import HeadRegistry


class BaseCLIPPipeline(nn.Module, ABC):
    """
    Base class for CLIP-based pipelines.
    Provides common functionality for feature extraction and head management.
    """
    
    def __init__(
        self,
        model: nn.Module,
        head: Optional[BaseHead] = None,
        assume_inputs_on_device: bool = True,
    ):
        """
        Args:
            model: Base CLIP model
            head: Head module to attach
            assume_inputs_on_device: Skip device moves if True
        """
        super().__init__()
        self.model = model
        self.head = head
        self.assume_inputs_on_device = assume_inputs_on_device
        self._device = None
    
    @property
    def device(self):
        """Cached device property."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def to(self, device):
        """Move module to device."""
        super().to(device)
        self._device = device
        return self
    
    def attach_head(self, head: BaseHead):
        """
        Attach a new head to the pipeline.
        
        Args:
            head: Head module to attach
        """
        self.head = head
        logging.info(f"Attached head: {head.__class__.__name__}")
    
    def detach_head(self):
        """Remove the current head."""
        self.head = None
        logging.info("Detached head")
    
    @abstractmethod
    def extract_features(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Extract features from inputs.
        
        Args:
            image: Image tensor
            text: Text tensor
        
        Returns:
            Dictionary with 'image' and/or 'text' features
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the pipeline.
        
        Args:
            image: Image tensor
            text: Text tensor
        
        Returns:
            Model outputs (depends on head type)
        """
        pass


class CLIPEndToEndPipeline(BaseCLIPPipeline):
    """
    Pipeline that takes raw inputs (images/text) and extracts features from CLIP.
    Suitable for fine-tuning scenarios where the CLIP model is also trained.
    """
    
    def __init__(
        self,
        model: nn.Module,
        head: BaseHead,
        image_layer_names: Optional[List[str]] = None,
        text_layer_names: Optional[List[str]] = None,
        ft_image_encoder: bool = True,
        ft_text_encoder: bool = True,
        assume_inputs_on_device: bool = True,
    ):
        """
        Args:
            model: Base CLIP model
            head: Head module to attach
            image_layer_names: Names of image layers to extract
            text_layer_names: Names of text layers to extract
            ft_image_encoder: Whether to finetune image encoder
            ft_text_encoder: Whether to finetune text encoder
            assume_inputs_on_device: Skip device moves if True
        """
        super().__init__(model=model, head=head, assume_inputs_on_device=assume_inputs_on_device)
        self.image_layer_names = image_layer_names
        self.text_layer_names = text_layer_names
        self.ft_image_encoder = ft_image_encoder
        self.ft_text_encoder = ft_text_encoder
        
        # Set requires_grad according to flags
        self.set_finetune(image_encoder=ft_image_encoder, text_encoder=ft_text_encoder)
    
    @property
    def temperature(self):
        """
        Get temperature from the head or base model if available.
        Returns a default value if neither has temperature.
        """
        # Then try base model's temperature attribute (for fine-tuned CLIP models)
        if self.model is not None and hasattr(self.model, 'temperature'):
            return self.model.temperature
        # Then try computing from logit_scale (standard CLIP)
        if self.model is not None and hasattr(self.model, 'logit_scale'):
            # CLIP models use logit_scale, where temperature = 1/exp(logit_scale)
            return 1.0 / self.model.logit_scale.exp()
        if self.head is not None and hasattr(self.head, 'temperature'):
            return self.head.temperature
        # Default CLIP temperature
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = 'cpu'
        return torch.tensor(0.07, device=device)
    
    def set_finetune(self, image_encoder: Optional[bool] = None, text_encoder: Optional[bool] = None):
        """
        Dynamically sets requires_grad for image and/or text encoder.
        
        Args:
            image_encoder: Whether to finetune image encoder
            text_encoder: Whether to finetune text encoder
        """
        if image_encoder is None:
            image_encoder = self.ft_image_encoder
        if text_encoder is None:
            text_encoder = self.ft_text_encoder

        if hasattr(self.model, "visual") and image_encoder is not None:
            for p in self.model.visual.parameters():
                p.requires_grad = image_encoder
            self.ft_image_encoder = image_encoder

        if hasattr(self.model, "transformer") and text_encoder is not None:
            for p in self.model.transformer.parameters():
                p.requires_grad = text_encoder
            self.ft_text_encoder = text_encoder
    
    def extract_features(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Extract features from raw inputs using CLIP model.
        
        Args:
            image: Image tensor
            text: Text tensor
        
        Returns:
            Dictionary with 'image' and/or 'text' features
        """
        from utils.align import extract_intermediate_features
        
        model_device = self.device
        feats = {}
        
        if image is not None:
            if not self.assume_inputs_on_device:
                image = image.to(model_device, non_blocking=True)
            feats['image'] = extract_intermediate_features(
                image, self.model, device=model_device,
                layer_names=self.image_layer_names, is_image=True,
                dtype=self.model.dtype
            )
        
        if text is not None:
            if not self.assume_inputs_on_device:
                text = text.to(model_device, non_blocking=True)
            feats['text'] = extract_intermediate_features(
                text, self.model, device=model_device,
                layer_names=self.text_layer_names, is_image=False,
                dtype=self.model.dtype
            )
        
        return feats
    
    def encode_image(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode image to embeddings."""
        feats = self.extract_features(image=image)
        if 'image' in feats and isinstance(self.head, BimodalHead):
            return self.head.encode_image(feats['image'])
        return None
    
    def encode_text(self, text: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode text to embeddings."""
        feats = self.extract_features(text=text)
        if 'text' in feats and isinstance(self.head, BimodalHead):
            return self.head.encode_text(feats['text'])
        return None
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through feature extraction and head.
        
        Args:
            image: Image tensor
            text: Text tensor
        
        Returns:
            Head outputs
        """
        if self.head is None:
            raise ValueError("No head attached to pipeline")
        
        # Extract features
        feats = self.extract_features(image=image, text=text)
        
        # Pass through head
        if isinstance(self.head, BimodalHead):
            return self.head(
                image_features=feats.get('image'),
                text_features=feats.get('text')
            )
        else:
            # Generic head - try to call with available features
            if 'image' in feats and 'text' in feats:
                return self.head(feats['image'], feats['text'])
            elif 'image' in feats:
                return self.head(feats['image'])
            elif 'text' in feats:
                return self.head(feats['text'])
            else:
                raise ValueError("No features extracted")


# Import nullcontext for Python 3.7+ compatibility
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    
    @contextmanager
    def nullcontext():
        yield
