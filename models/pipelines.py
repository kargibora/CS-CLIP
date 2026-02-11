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

from .base_heads import BaseHead, BimodalHead, ScoringHead
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


class CLIPFeaturePipeline(BaseCLIPPipeline):
    """
    Pipeline that works with pre-extracted CLIP features.
    Suitable for LabCLIP-style models where features are cached.
    """
    
    def __init__(
        self,
        head: BaseHead,
        assume_inputs_on_device: bool = True,
    ):
        """
        Args:
            head: Head module to attach
            assume_inputs_on_device: Skip device moves if True
        """
        # No base model needed - works with features directly
        super().__init__(model=None, head=head, assume_inputs_on_device=assume_inputs_on_device)
    
    @property
    def temperature(self):
        """
        Get temperature from the head if available.
        Returns a default value if head doesn't have temperature.
        """
        if self.head is not None and hasattr(self.head, 'temperature'):
            return self.head.temperature
        # Default CLIP temperature
        # Standard CLIP uses temperature = 0.07, which corresponds to logit_scale = ln(1/0.07) ≈ 2.6593
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # No parameters in this module, use CPU
            device = 'cpu'
        return torch.tensor(0.07, device=device)
    
    def extract_features(
        self,
        image: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        text: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        In this pipeline, inputs are already features, so just pass them through.
        
        Args:
            image: Image features
            text: Text features
        
        Returns:
            Dictionary with 'image' and/or 'text' features
        """
        feats = {}
        if image is not None:
            feats['image'] = image
        if text is not None:
            feats['text'] = text
        return feats
    
    def encode_image(
        self, 
        image_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Encode image features through the head.
        
        Args:
            image_features: Pre-extracted image features (tensor or dict)
        
        Returns:
            Aligned image embeddings
        """
        if self.head is None:
            raise ValueError("No head attached to pipeline")
        
        # For BimodalHead, use encode_image method
        if hasattr(self.head, 'encode_image'):
            return self.head.encode_image(image_features)
        
        # For other heads, extract final feature and normalize
        # (to match FlexibleCLIPMultiLayerAlignment behavior)
        if isinstance(image_features, dict):
            # Return the last/final layer if it's a dict
            if 'final' in image_features:
                feat = image_features['final']
            else:
                feat = list(image_features.values())[-1]
        else:
            feat = image_features
        
        # Normalize for fp16/bf16 stability (matches FlexibleCLIPMultiLayerAlignment)
        return F.normalize(feat, dim=-1, eps=1e-6)
    
    def encode_text(
        self, 
        text_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Encode text features through the head.
        
        Args:
            text_features: Pre-extracted text features (tensor or dict)
        
        Returns:
            Aligned text embeddings
        """
        if self.head is None:
            raise ValueError("No head attached to pipeline")
        
        # For BimodalHead, use encode_text method
        if hasattr(self.head, 'encode_text'):
            return self.head.encode_text(text_features)
        
        # For other heads, extract final feature and normalize
        # (to match FlexibleCLIPMultiLayerAlignment behavior)
        if isinstance(text_features, dict):
            # Return the last/final layer if it's a dict
            if 'final' in text_features:
                feat = text_features['final']
            else:
                feat = list(text_features.values())[-1]
        else:
            feat = text_features
        
        # Normalize for fp16/bf16 stability (matches FlexibleCLIPMultiLayerAlignment)
        return F.normalize(feat, dim=-1, eps=1e-6)
    
    def forward(
        self,
        image_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        text_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None
    ):
        """
        Forward pass through the head.
        
        Args:
            image_features: Image features
            text_features: Text features
        
        Returns:
            Head outputs
        """
        if self.head is None:
            raise ValueError("No head attached to pipeline")
        
        # Handle different head types
        if isinstance(self.head, BimodalHead):
            return self.head(image_features=image_features, text_features=text_features)
        elif isinstance(self.head, ScoringHead):
            if image_features is None or text_features is None:
                raise ValueError("ScoringHead requires both image and text features")
            return self.head(image_features=image_features, text_features=text_features)
        else:
            # Generic head - try to call with available features
            if image_features is not None and text_features is not None:
                return self.head(image_features, text_features)
            elif image_features is not None:
                return self.head(image_features)
            elif text_features is not None:
                return self.head(text_features)
            else:
                raise ValueError("No features provided")


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
    
    def extract_patch_and_token_features(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract patch-level and token-level features for TCA.
        
        Args:
            image: Image tensor [B, C, H, W]
            text: Text tensor [B, L] (tokenized text)
            
        Returns:
            Dictionary with:
                - image_global: [B, D] global image features (or None)
                - image_patches: [B, P, D] patch-level features (or None)
                - text_global: [B, D] global text features (or None)
                - text_tokens: [B, L, D] token-level features (or None)
                - attention_mask: [B, L] mask for valid text tokens (or None)
        """
        model_device = self.device
        result = {
            'image_global': None,
            'image_patches': None,
            'text_global': None,
            'text_tokens': None,
            'attention_mask': None,
        }
        
        # Extract image features
        if image is not None:
            if not self.assume_inputs_on_device:
                image = image.to(model_device, non_blocking=True)
            
            # Extract patches from vision transformer
            if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
                # CRITICAL FIX: Force training mode to avoid batch norm collapsing diversity
                # Eval mode batch norm uses running stats which can cause identical outputs
                was_training = self.model.visual.transformer.training
                self.model.visual.transformer.train()
                
                # Convolutional patch extraction
                x = self.model.visual.conv1(image)  # [B, D, H', W']
                
                # DEBUG: Check after conv1
                if not hasattr(self, '_debug_conv1_logged'):
                    if x.shape[0] > 1:
                        conv1_diff = torch.norm(x[0] - x[1]).item()
                        logging.info(f"[Pipeline CLIPEndToEnd] After conv1 - diff between first two: {conv1_diff:.6f}")
                    self._debug_conv1_logged = True
                
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, P, D]
                
                # Add class token
                # CRITICAL: Use .repeat() instead of .expand() to create independent copies
                # .expand() creates a view that shares memory → all samples become identical!
                class_emb = self.model.visual.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
                
                # DEBUG: Check class embedding
                if not hasattr(self, '_debug_class_emb_logged'):
                    if class_emb.shape[0] > 1:
                        cls_diff = torch.norm(class_emb[0] - class_emb[1]).item()
                        logging.info(f"[Pipeline CLIPEndToEnd] Class embeddings - diff between first two: {cls_diff:.6f}")
                        logging.info(f"[Pipeline CLIPEndToEnd] Class embedding is_contiguous: {class_emb.is_contiguous()}")
                        logging.info(f"[Pipeline CLIPEndToEnd] Class embedding data_ptr match: {class_emb[0].data_ptr() == class_emb[1].data_ptr()}")
                    self._debug_class_emb_logged = True
                
                x = torch.cat([class_emb, x], dim=1)  # [B, P+1, D]
                
                # DEBUG: Check after cat
                if not hasattr(self, '_debug_cat_logged'):
                    if x.shape[0] > 1:
                        cat_diff = torch.norm(x[0, 0, :] - x[1, 0, :]).item()  # CLS position
                        logging.info(f"[Pipeline CLIPEndToEnd] After cat (CLS) - diff between first two: {cat_diff:.6f}")
                    self._debug_cat_logged = True
                
                # Add positional embedding
                x = x + self.model.visual.positional_embedding.to(x.dtype)
                x = self.model.visual.ln_pre(x)
                
                # Pass through transformer
                x = x.permute(1, 0, 2)  # [P+1, B, D]
                
                # DEBUG: Check before transformer - PATCHES not just CLS!
                if not hasattr(self, '_debug_before_trans_logged'):
                    if x.shape[1] > 1:  # Check batch dimension (now at dim 1)
                        cls_diff = torch.norm(x[0, 0, :] - x[0, 1, :]).item()  # CLS token of first two samples
                        patch_diff = torch.norm(x[1, 0, :] - x[1, 1, :]).item()  # First PATCH of first two samples
                        logging.info(f"[Pipeline CLIPEndToEnd] Before transformer - CLS diff: {cls_diff:.6f}, PATCH[1] diff: {patch_diff:.6f}")
                        logging.info(f"[Pipeline CLIPEndToEnd] Model.visual.transformer.training: {self.model.visual.transformer.training}")
                    self._debug_before_trans_logged = True
                
                x = self.model.visual.transformer(x)
                
                # DEBUG: Check after transformer
                if not hasattr(self, '_debug_after_trans_logged'):
                    if x.shape[1] > 1:  # Check batch dimension (still at dim 1)
                        cls_diff = torch.norm(x[0, 0, :] - x[0, 1, :]).item()  # CLS token of first two samples  
                        patch_diff = torch.norm(x[1, 0, :] - x[1, 1, :]).item()  # First PATCH of first two samples
                        logging.info(f"[Pipeline CLIPEndToEnd] After transformer - CLS diff: {cls_diff:.6f}, PATCH[1] diff: {patch_diff:.6f}")
                    self._debug_after_trans_logged = True
                
                x = x.permute(1, 0, 2)  # [B, P+1, D]
                
                # Apply layer norm
                x = self.model.visual.ln_post(x)
                
                # DEBUG: Check after ln_post
                if not hasattr(self, '_debug_after_ln_logged'):
                    if x.shape[0] > 1:
                        ln_diff = torch.norm(x[0, 0, :] - x[1, 0, :]).item()  # CLS token
                        logging.info(f"[Pipeline CLIPEndToEnd] After ln_post (CLS) - diff between first two: {ln_diff:.6f}")
                    self._debug_after_ln_logged = True
                
                # Extract global (CLS) and patch features
                result['image_global'] = x[:, 0, :]  # [B, D_vision]
                result['image_patches'] = x[:, 1:, :]  # [B, P, D_vision] - keep in native space!
                
                # Project ONLY global features to output space
                if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
                    # DEBUG: Check before projection
                    if not hasattr(self, '_debug_before_proj_logged'):
                        if result['image_global'].shape[0] > 1:
                            before_proj_diff = torch.norm(result['image_global'][0] - result['image_global'][1]).item()
                            logging.info(f"[Pipeline CLIPEndToEnd] Before projection - diff between first two: {before_proj_diff:.6f}")
                        self._debug_before_proj_logged = True
                    
                    result['image_global'] = result['image_global'] @ self.model.visual.proj
                    
                    # DEBUG: Check after projection
                    if not hasattr(self, '_debug_after_proj_logged'):
                        if result['image_global'].shape[0] > 1:
                            after_proj_diff = torch.norm(result['image_global'][0] - result['image_global'][1]).item()
                            logging.info(f"[Pipeline CLIPEndToEnd] After projection - diff between first two: {after_proj_diff:.6f}")
                        self._debug_after_proj_logged = True
                    # NOTE: We do NOT project patches - TCA head will handle dimension mapping
                
                # Restore original training mode
                self.model.visual.transformer.train(was_training)
        
        # Extract text features
        if text is not None:
            if not self.assume_inputs_on_device:
                text = text.to(model_device, non_blocking=True)
            
            # Extract tokens from text transformer
            if hasattr(self.model, 'transformer'):
                # Embed tokens
                x = self.model.token_embedding(text).type(self.model.dtype)  # [B, L, D]
                
                # Add positional embedding
                x = x + self.model.positional_embedding.type(self.model.dtype)
                
                # Pass through transformer
                x = x.permute(1, 0, 2)  # [L, B, D]
                x = self.model.transformer(x)
                x = x.permute(1, 0, 2)  # [B, L, D]
                
                # Apply layer norm
                x = self.model.ln_final(x).type(self.model.dtype)
                
                # Extract token features
                result['text_tokens'] = x  # [B, L, D_text] - keep in native space!
                
                # Create attention mask (1 for real tokens, 0 for padding)
                # Padding tokens are 0 in CLIP tokenization
                result['attention_mask'] = (text != 0).long()  # [B, L]
                
                # Extract global feature (EOS token)
                # Take features from the eot token (highest index with value)
                eot_indices = text.argmax(dim=-1)  # [B]
                result['text_global'] = x[torch.arange(x.shape[0]), eot_indices]  # [B, D_text]
                
                # Project ONLY global features to output space
                if hasattr(self.model, 'text_projection') and self.model.text_projection is not None:
                    result['text_global'] = result['text_global'] @ self.model.text_projection
                    # NOTE: We do NOT project tokens - TCA head will handle dimension mapping
        
        # Package image features into combined 'image' dict for heads that expect it
        # This supports both TQA (needs global + patches) and standard heads (need just global)
        if result.get('image_global') is not None:
            result['image'] = {
                'global': result['image_global'],
                'patches': result.get('image_patches'),  # May be None for some pipelines
            }
        
        # Package text features into combined 'text' dict for heads that expect it
        if result.get('text_global') is not None:
            result['text'] = {
                'global': result['text_global'],
                'tokens': result.get('text_tokens'),  # May be None for some pipelines
                'attention_mask': result.get('attention_mask'),
            }
        
        return result
    
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
        elif isinstance(self.head, ScoringHead):
            if 'image' not in feats or 'text' not in feats:
                raise ValueError("ScoringHead requires both image and text inputs")
            return self.head(
                image_features=feats['image'],
                text_features=feats['text']
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


class TokenConditionedPipeline(BaseCLIPPipeline):
    """
    Pipeline for Token-Conditioned Alignment (TCA).
    
    This pipeline:
    1. Extracts patch-level image features and token-level text features
    2. Freezes CLIP encoders (no caching possible)
    3. Only trains alignment matrices in the head
    
    Cannot use cached embeddings since it needs intermediate transformer outputs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        head: Optional[BaseHead] = None,
        freeze_clip: bool = True,
        assume_inputs_on_device: bool = True,
    ):
        """
        Args:
            model: Base CLIP model with access to intermediate features
            head: TCA head module
            freeze_clip: Whether to freeze CLIP encoders
            assume_inputs_on_device: Skip device moves if True
        """
        super().__init__(model, head, assume_inputs_on_device)
        
        self.freeze_clip = freeze_clip
        
        if freeze_clip:
            # Freeze all CLIP parameters
            for param in self.model.parameters():
                param.requires_grad = False
            logging.info("Frozen CLIP encoders for TCA")
    
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
        # First try head
        if self.head is not None and hasattr(self.head, 'temperature'):
            return self.head.temperature

        # Default CLIP temperature
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = 'cpu'
        return torch.tensor(0.07, device=device)
    
    def extract_patch_and_token_features(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global, patch-level, and token-level features.
        
        Args:
            image: Image tensor [B, C, H, W]
            text: Text tensor [B, L]
            
        Returns:
            Dictionary with:
                - image_global: [B, D] global image features
                - text_global: [B, D] global text features
                - image_patches: [B, P, D] patch-level features
                - text_tokens: [B, L, D] token-level features
        """
        features = {}
        
        if image is not None:
            # Debug: Check if images are different
            if not hasattr(self, '_debug_images_logged'):
                import logging
                if image.shape[0] > 1:
                    img_diff = torch.norm(image[0] - image[1]).item()
                    logging.info(f"[Pipeline] Input images - difference between first two: {img_diff:.4f}")
                    logging.info(f"[Pipeline] Input images - shape: {image.shape}, dtype: {image.dtype}")
                    logging.info(f"[Pipeline] Input images - min: {image.min().item():.4f}, max: {image.max().item():.4f}")
                    logging.info(f"[Pipeline] Model training mode: {self.model.training}")
                    logging.info(f"[Pipeline] Model.visual training mode: {self.model.visual.training}")
                self._debug_images_logged = True
            
            # Extract image features
            if hasattr(self.model, 'encode_image_with_patches'):
                # Custom method that returns both global and patches
                img_global, img_patches = self.model.encode_image_with_patches(image)
            elif hasattr(self.model.visual, 'transformer'):
                # OpenCLIP/CLIP vision transformer
                x = self.model.visual.conv1(image)  # [B, D, H', W']
                x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, D, P]
                x = x.permute(0, 2, 1)  # [B, P, D]
                
                # CRITICAL: Use .repeat() instead of .expand() to create independent copies
                # .expand() creates a view that shares memory → all samples become identical!
                class_emb = self.model.visual.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
                x = torch.cat([class_emb, x], dim=1)  # [B, P+1, D]
                
                x = x + self.model.visual.positional_embedding.to(x.dtype)
                x = self.model.visual.ln_pre(x)
                
                x = x.permute(1, 0, 2)  # [P+1, B, D]
                x = self.model.visual.transformer(x)
                x = x.permute(1, 0, 2)  # [B, P+1, D]
                
                # Debug: Check transformer output
                if not hasattr(self, '_debug_transformer_logged'):
                    import logging
                    if x.shape[0] > 1:
                        trans_diff = torch.norm(x[0, 0, :] - x[1, 0, :]).item()  # CLS token
                        logging.info(f"[Pipeline] After transformer - CLS diff between first two: {trans_diff:.6f}")
                    self._debug_transformer_logged = True
                
                img_global = self.model.visual.ln_post(x[:, 0, :])  # [B, D]
                if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
                    img_global = img_global @ self.model.visual.proj
                
                # Debug: Check if features are different
                if not hasattr(self, '_debug_img_features_logged'):
                    import logging
                    if img_global.shape[0] > 1:
                        feat_diff = torch.norm(img_global[0] - img_global[1]).item()
                        logging.info(f"[Pipeline] Image features - difference between first two: {feat_diff:.4f}")
                        logging.info(f"[Pipeline] Image features - shape: {img_global.shape}")
                    self._debug_img_features_logged = True
                
                img_patches = self.model.visual.ln_post(x[:, 1:, :])  # [B, P, D]
            else:
                # Fallback: use standard encode_image
                img_global = self.model.encode_image(image)
                img_patches = None
                logging.warning("Could not extract patch features, using global only")
            
            features['image_global'] = img_global
            if img_patches is not None:
                features['image_patches'] = img_patches
        
        if text is not None:
            # Extract text features
            if hasattr(self.model, 'encode_text_with_tokens'):
                # Custom method that returns both global and tokens
                txt_global, txt_tokens = self.model.encode_text_with_tokens(text)
            elif hasattr(self.model, 'transformer'):
                # OpenCLIP/CLIP text transformer
                x = self.model.token_embedding(text).type(self.model.dtype)  # [B, L, D]
                x = x + self.model.positional_embedding.type(self.model.dtype)
                x = x.permute(1, 0, 2)  # [L, B, D]
                x = self.model.transformer(x)
                x = x.permute(1, 0, 2)  # [B, L, D]
                x = self.model.ln_final(x).type(self.model.dtype)
                
                # Get global: take features at EOT token
                txt_global = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                if hasattr(self.model, 'text_projection') and self.model.text_projection is not None:
                    txt_global = txt_global @ self.model.text_projection
                
                txt_tokens = x  # [B, L, D] all token features
            else:
                # Fallback: use standard encode_text
                txt_global = self.model.encode_text(text)
                txt_tokens = None
                logging.warning("Could not extract token features, using global only")
            
            features['text_global'] = txt_global
            if txt_tokens is not None:
                features['text_tokens'] = txt_tokens
        
        return features
    
    def extract_features(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Extract features including patches and tokens.
        
        Returns:
            Dictionary with all feature types
        """
        if self.freeze_clip:
            with torch.no_grad():
                return self.extract_patch_and_token_features(image, text)
        else:
            return self.extract_patch_and_token_features(image, text)
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through TCA pipeline.
        
        Args:
            image: Image tensor
            text: Text tensor
            **kwargs: Additional arguments for head
        
        Returns:
            Head outputs with caption-conditioned features
        """
        if self.head is None:
            raise ValueError("No head attached to pipeline")
        
        # Extract all features
        feats = self.extract_features(image=image, text=text)
        
        # Pass through TCA head
        return self.head(
            image_features=feats.get('image_global'),
            text_features=feats.get('text_global'),
            image_patches=feats.get('image_patches'),
            text_tokens=feats.get('text_tokens'),
            **kwargs
        )
    
    def encode_image(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode image to caption-conditioned embedding (requires paired text)."""
        raise NotImplementedError(
            "TCA requires both image and text. Use forward() with both inputs."
        )
    
    def encode_text(self, text: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode text to aligned embedding."""
        feats = self.extract_features(text=text)
        if self.head is not None:
            return self.head.align_text(feats['text_global'])
        return feats['text_global']


class TextQueryAggregatorPipeline(BaseCLIPPipeline):
    """
    Pipeline for Text-Query Aggregation (TQA).
    
    Implements the advisor's suggestion:
        "Add a layer pre projection of the CLS token and inject a mini aggregator,
        probably an attention head between the keys of the patches and query of the text embedding."
    
    In CLIP terms:
        v_cls' = v_cls + Attn(q=t, k=v_patches, v=v_patches)
    
    Then:
        - Project v_cls' to image space
        - Project text as usual
        - Do InfoNCE/contrastive loss
    
    This pipeline:
    1. Extracts global and patch-level image features from CLIP's vision transformer
    2. Extracts text features from CLIP's text transformer
    3. Uses the TextQueryAggregatorHead to create caption-conditioned image embeddings
    4. Supports standard InfoNCE contrastive training
    """
    
    def __init__(
        self,
        model: nn.Module,
        head: Optional[BaseHead] = None,
        freeze_clip: bool = True,
        freeze_text_encoder: bool = True,
        freeze_vision_encoder: bool = True,
        assume_inputs_on_device: bool = True,
    ):
        """
        Args:
            model: Base CLIP model with access to intermediate features
            head: TextQueryAggregatorHead or compatible head
            freeze_clip: Whether to freeze entire CLIP model (overrides individual settings)
            freeze_text_encoder: Whether to freeze text encoder
            freeze_vision_encoder: Whether to freeze vision encoder
            assume_inputs_on_device: Skip device moves if True
        """
        super().__init__(model, head, assume_inputs_on_device)
        
        self.freeze_clip = freeze_clip
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_vision_encoder = freeze_vision_encoder
        
        if freeze_clip:
            # Freeze all CLIP parameters
            for param in self.model.parameters():
                param.requires_grad = False
            logging.info("[TQA Pipeline] Frozen entire CLIP model")
        else:
            # Freeze individual components
            if freeze_vision_encoder and hasattr(self.model, 'visual'):
                for param in self.model.visual.parameters():
                    param.requires_grad = False
                logging.info("[TQA Pipeline] Frozen vision encoder")
            
            if freeze_text_encoder and hasattr(self.model, 'transformer'):
                for param in self.model.transformer.parameters():
                    param.requires_grad = False
                logging.info("[TQA Pipeline] Frozen text encoder")
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"[TQA Pipeline] Trainable: {trainable_params:,} / {total_params:,} parameters")
    
    @property
    def temperature(self):
        """Get temperature from head or model."""
        if self.head is not None and hasattr(self.head, 'temperature'):
            return self.head.temperature
        if self.model is not None and hasattr(self.model, 'logit_scale'):
            return 1.0 / self.model.logit_scale.exp()
        return torch.tensor(0.07, device=self.device)
    
    def extract_patch_features(
        self,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global CLS and patch-level features from vision transformer.
        
        Args:
            image: Image tensor [B, C, H, W]
            
        Returns:
            Dict with:
                - 'global': CLS token embedding [B, D_vision]
                - 'patches': Patch embeddings [B, P, D_vision]
        """
        model_device = self.device
        if not self.assume_inputs_on_device:
            image = image.to(model_device, non_blocking=True)
        
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
            visual = self.model.visual
            
            # Patch embedding
            x = visual.conv1(image)  # [B, D, H', W']
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, P, D]
            
            # Add class token
            class_emb = visual.class_embedding.to(x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([class_emb, x], dim=1)  # [B, P+1, D]
            
            # Add positional embedding
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            
            # Pass through transformer
            x = x.permute(1, 0, 2)  # [P+1, B, D]
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)  # [B, P+1, D]
            
            # Layer norm
            x = visual.ln_post(x)
            
            # Split into global and patches
            global_feat = x[:, 0, :]  # [B, D_vision]
            patch_feats = x[:, 1:, :]  # [B, P, D_vision]
            
            # Project global feature to output space (for standard CLIP compatibility)
            if hasattr(visual, 'proj') and visual.proj is not None:
                global_feat_projected = global_feat @ visual.proj
            else:
                global_feat_projected = global_feat
            
            return {
                'global': global_feat_projected,
                'global_unprojected': global_feat,  # Keep native space for TQA
                'patches': patch_feats,  # Keep in native space
            }
        else:
            # Fallback for models without accessible internals
            raise NotImplementedError(
                "TQA Pipeline requires CLIP model with accessible vision transformer internals"
            )
    
    def extract_text_features(
        self,
        text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global text features.
        
        Args:
            text: Tokenized text [B, L]
            
        Returns:
            Dict with:
                - 'global': Text embedding [B, D_text]
        """
        model_device = self.device
        if not self.assume_inputs_on_device:
            text = text.to(model_device, non_blocking=True)
        
        if hasattr(self.model, 'transformer'):
            # Token embedding
            x = self.model.token_embedding(text).type(self.model.dtype)  # [B, L, D]
            x = x + self.model.positional_embedding.type(self.model.dtype)
            
            # Transformer
            x = x.permute(1, 0, 2)  # [L, B, D]
            x = self.model.transformer(x)
            x = x.permute(1, 0, 2)  # [B, L, D]
            
            # Layer norm
            x = self.model.ln_final(x).type(self.model.dtype)
            
            # Extract EOS token feature
            eot_indices = text.argmax(dim=-1)  # [B]
            global_feat = x[torch.arange(x.shape[0], device=x.device), eot_indices]  # [B, D_text]
            
            # Project to output space
            if hasattr(self.model, 'text_projection') and self.model.text_projection is not None:
                global_feat = global_feat @ self.model.text_projection
            
            return {'global': global_feat}
        else:
            raise NotImplementedError(
                "TQA Pipeline requires CLIP model with accessible text transformer"
            )
    
    def extract_features(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Extract features from image and/or text.
        
        Returns:
            Dict with 'image' and/or 'text' sub-dicts containing features
        """
        feats = {}
        
        context = torch.no_grad() if self.freeze_clip else nullcontext()
        
        with context:
            if image is not None:
                feats['image'] = self.extract_patch_features(image)
            
            if text is not None:
                feats['text'] = self.extract_text_features(text)
        
        return feats
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass through TQA pipeline.
        
        Args:
            image: Image tensor [B, C, H, W]
            text: Tokenized text [B, L]
            return_dict: Whether to return a dict
            **kwargs: Additional arguments for head
            
        Returns:
            Dict with 'image_embeds', 'text_embeds', 'logit_scale', etc.
        """
        if self.head is None:
            raise ValueError("No head attached to pipeline")
        
        # Extract features
        feats = self.extract_features(image=image, text=text)
        
        # Prepare inputs for head
        image_features = feats.get('image', {})
        text_features = feats.get('text', {})
        
        # For TQA head, we need to pass the global and patches separately
        # and use the text to condition the image
        return self.head(
            image_features=image_features,
            text_features=text_features,
            image_patches=image_features.get('patches'),
            return_dict=return_dict,
            **kwargs
        )
    
    def encode_image(
        self,
        image: torch.Tensor,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode image to embedding.
        
        If text is provided, creates caption-conditioned embedding.
        Otherwise, returns standard CLIP image embedding.
        
        Args:
            image: Image tensor
            text: Optional text for conditioning
            
        Returns:
            Image embedding
        """
        image_feats = self.extract_patch_features(image)
        
        if text is not None and self.head is not None:
            text_feats = self.extract_text_features(text)
            return self.head.encode_image(image_feats, text_features=text_feats['global'])
        elif self.head is not None:
            return self.head.encode_image(image_feats)
        else:
            return F.normalize(image_feats['global'], dim=-1)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to embedding."""
        text_feats = self.extract_text_features(text)
        
        if self.head is not None:
            return self.head.encode_text(text_feats)
        else:
            return F.normalize(text_feats['global'], dim=-1)
    
    def encode_multimodal(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode image with text conditioning and return both embeddings.
        
        This is the core TQA operation for compositional benchmarks:
            v_cls' = v_cls + Attn(q=text, k=patches, v=patches)
        
        Args:
            image: Image tensor [B, C, H, W]
            text_tokens: Tokenized text [B, L]
            
        Returns:
            Dict with:
                - 'image_embeds': Caption-conditioned image embeddings [B, D]
                - 'text_embeds': Text embeddings [B, D]
                - 'base_image_embeds': Unconditioned image embeddings [B, D]
        """
        # Extract features
        image_feats = self.extract_patch_features(image)
        text_feats = self.extract_text_features(text_tokens)
        
        if self.head is not None:
            # Get caption-conditioned image embedding (TQA cross-attention)
            conditioned_image_emb = self.head.encode_image(
                image_feats, text_features=text_feats['global']
            )
            
            # Get unconditioned image embedding (for comparison)
            unconditioned_image_emb = self.head.encode_image(image_feats)
            
            # Get text embedding
            text_emb = self.head.encode_text(text_feats)
            
            return {
                'image_embeds': conditioned_image_emb,
                'text_embeds': text_emb,
                'base_image_embeds': unconditioned_image_emb,
            }
        else:
            # Fallback without head
            return {
                'image_embeds': F.normalize(image_feats['global'], dim=-1),
                'text_embeds': F.normalize(text_feats['global'], dim=-1),
                'base_image_embeds': F.normalize(image_feats['global'], dim=-1),
            }


# Import nullcontext for Python 3.7+ compatibility
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    
    @contextmanager
    def nullcontext():
        yield


def create_pipeline_from_config(config: Dict) -> BaseCLIPPipeline:
    """
    Factory function to create a pipeline from configuration.
    
    Config format:
        {
            'type': 'feature' or 'end_to_end',
            'head': {
                'type': 'bimodal' or 'scoring' or 'alignment',
                'name': 'flexible_multi_layer' or 'cross_modal_mlp' etc.,
                'params': {...}
            },
            'model': <CLIP model instance> (only for end_to_end),
            'image_layer_names': [...] (only for end_to_end),
            'text_layer_names': [...] (only for end_to_end),
            'ft_image_encoder': bool (only for end_to_end),
            'ft_text_encoder': bool (only for end_to_end),
        }
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Instantiated pipeline
    """
    pipeline_type = config.get('type', 'feature')
    head_config = config.get('head', {})
    
    # Create head from config
    head = HeadRegistry.create_from_config(head_config)
    
    if pipeline_type == 'feature':
        return CLIPFeaturePipeline(
            head=head,
            assume_inputs_on_device=config.get('assume_inputs_on_device', True)
        )
    elif pipeline_type == 'end_to_end':
        model = config.get('model')
        if model is None:
            raise ValueError("'model' must be provided for end_to_end pipeline")
        
        return CLIPEndToEndPipeline(
            model=model,
            head=head,
            image_layer_names=config.get('image_layer_names'),
            text_layer_names=config.get('text_layer_names'),
            ft_image_encoder=config.get('ft_image_encoder', True),
            ft_text_encoder=config.get('ft_text_encoder', True),
            assume_inputs_on_device=config.get('assume_inputs_on_device', True)
        )
    
    elif pipeline_type == "token_conditioned":
        # TCA pipeline requires frozen CLIP model + patch/token extraction
        model = config.get('model')
        if model is None:
            raise ValueError("'model' must be provided for token_conditioned pipeline")
        
        return TokenConditionedPipeline(
            model=model,
            head=head,
            freeze_clip=config.get('freeze_clip', True),
            assume_inputs_on_device=config.get('assume_inputs_on_device', True)
        )
    
    elif pipeline_type == "text_query_aggregator":
        # TQA pipeline: text queries image patches to create caption-conditioned embeddings
        model = config.get('model')
        if model is None:
            raise ValueError("'model' must be provided for text_query_aggregator pipeline")
        
        return TextQueryAggregatorPipeline(
            model=model,
            head=head,
            freeze_clip=config.get('freeze_clip', True),
            freeze_text_encoder=config.get('freeze_text_encoder', True),
            freeze_vision_encoder=config.get('freeze_vision_encoder', True),
            assume_inputs_on_device=config.get('assume_inputs_on_device', True)
        )
    
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
