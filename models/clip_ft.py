"""
Fine-tunable CLIP wrapper with multi-layer alignment support.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict

from utils.align import extract_intermediate_features
from .labclip import FlexibleCLIPCrossModalHead, FlexibleCLIPMultiLayerAlignment


class CLIPMultiLayerFTAlignment(nn.Module):
    """
    Fine-tunable CLIP wrapper that aligns multi-layer features via a FlexibleCLIPMultiLayerAlignment head.
    Supports both standard CLIP alignment and LabCLIP scoring via use_cross_modal_head flag.
    """
    def __init__(
        self,
        model: nn.Module,
        image_layer_names: Optional[List[str]],
        text_layer_names: Optional[List[str]],
        image_layer_dims: Optional[List[int]],
        text_layer_dims: Optional[List[int]],
        align_cfg: Optional[dict] = None,
        aggregator_cfg: Optional[dict] = None,
        ft_image_encoder: bool = True,
        ft_text_encoder: bool = True,
        ft_start_epoch: int = 0,
        align_image: bool = False,
        align_text: bool = True,
        assume_inputs_on_device: bool = True,
        init_temperature: Optional[float] = None,
        freeze_temperature: bool = False,
        use_cross_modal_head: bool = False,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        embed_dim: Optional[int] = None,
        use_alignment: bool = True,
    ):
        """
        Args:
            model: Base CLIP model
            image_layer_names: Names of image layers to extract
            text_layer_names: Names of text layers to extract
            image_layer_dims: Dimensions of image layers
            text_layer_dims: Dimensions of text layers
            align_cfg: Configuration for alignment heads
            aggregator_cfg: Configuration for aggregators
            ft_image_encoder: Whether to finetune image encoder
            ft_text_encoder: Whether to finetune text encoder
            ft_start_epoch: Epoch to start finetuning
            align_image: Whether to align image features
            align_text: Whether to align text features
            assume_inputs_on_device: Skip device moves if True
            init_temperature: Initial temperature (None = detect from model)
            freeze_temperature: Whether to freeze temperature
            use_cross_modal_head: Use LabCLIP scoring instead of standard alignment
            mlp_hidden_dims: Hidden dimensions for LabCLIP MLP
            dropout: Dropout rate for LabCLIP MLP
            embed_dim: Embedding dimension for LabCLIP
            use_alignment: Whether to use alignment aggregators in LabCLIP
        """
        super().__init__()
        self.model = model
        self.align_image = align_image
        self.align_text = align_text
        self.assume_inputs_on_device = assume_inputs_on_device
        self.use_cross_modal_head = use_cross_modal_head

        self.ft_image_encoder = ft_image_encoder
        self.ft_text_encoder  = ft_text_encoder
        self.ft_start_epoch   = ft_start_epoch

        # Set (un)freezing according to flags
        self.set_finetune(image_encoder=ft_image_encoder, text_encoder=ft_text_encoder)

        logging.info(
            "CLIPMultiLayerFTAlignment: ft_image_encoder=%s, ft_text_encoder=%s, "
            "align_image=%s, align_text=%s, use_cross_modal_head=%s",
            ft_image_encoder, ft_text_encoder, align_image, align_text, use_cross_modal_head
        )

        n_trainable_img = sum(p.requires_grad for p in getattr(model, "visual", nn.Module()).parameters())
        n_trainable_txt = sum(p.requires_grad for p in getattr(model, "transformer", nn.Module()).parameters())
        logging.info("Trainable image params: %d", n_trainable_img)
        logging.info("Trainable text params: %d", n_trainable_txt)

        # Sanity checks (only if encoders exist)
        if ft_image_encoder and hasattr(self.model, "visual"):
            assert any(p.requires_grad for p in self.model.visual.parameters()), (
                "Image encoder set to finetune, but no parameters are trainable."
            )
        if ft_text_encoder and hasattr(self.model, "transformer"):
            assert any(p.requires_grad for p in self.model.transformer.parameters()), (
                "Text encoder set to finetune, but no parameters are trainable."
            )

        # Flexible alignment head (holds the single temperature `t`)
        if use_cross_modal_head:
            # Use LabCLIP scoring head
            self.align_head = FlexibleCLIPCrossModalHead(
                image_layer_dims=image_layer_dims,
                text_layer_dims=text_layer_dims,
                align_cfg=align_cfg,
                aggregator_cfg=aggregator_cfg,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                image_layer_names=image_layer_names,
                text_layer_names=text_layer_names,
                embed_dim=embed_dim,
                use_alignment=use_alignment,
            )
        else:
            # Use standard CLIP alignment head
            self.align_head = FlexibleCLIPMultiLayerAlignment(
                image_layer_dims=image_layer_dims,
                text_layer_dims=text_layer_dims,
                align_cfg=align_cfg,
                aggregator_cfg=aggregator_cfg,
                align_image=align_image,
                align_text=align_text,
                image_layer_names=image_layer_names,
                text_layer_names=text_layer_names,
            )

        self.image_layer_names = image_layer_names
        self.text_layer_names  = text_layer_names

        # Temperature is only relevant for standard CLIP, not LabCLIP
        if not use_cross_modal_head:
            if init_temperature is None:
                init_logit_scale = self._detect_pretrained_logit_scale()
            else:
                init_logit_scale = torch.log(torch.tensor(1.0 / init_temperature))
            
            # Initialize temperature parameter
            if freeze_temperature:
                # Keep original temperature frozen
                self.register_buffer('logit_scale', init_logit_scale)
            else:
                # Allow temperature to be learned during finetuning
                self.logit_scale = nn.Parameter(init_logit_scale)
        else:
            # For LabCLIP, we don't need temperature/logit_scale
            self.register_buffer('logit_scale', torch.tensor(0.0))  # Dummy value

    def _detect_pretrained_logit_scale(self) -> torch.Tensor:
        """Extract logit_scale directly from pretrained CLIP model."""
        print(f"Model attributes: {[attr for attr in dir(self.model) if 'scale' in attr.lower() or 'temp' in attr.lower()]}")
        
        # Common locations for logit_scale in CLIP models
        if hasattr(self.model, 'logit_scale'):
            # OpenAI CLIP format - use logit_scale directly
            logit_scale_value = self.model.logit_scale.item() if hasattr(self.model.logit_scale, 'item') else self.model.logit_scale
            print(f"Found logit_scale: {logit_scale_value}")
            temperature = 1.0 / torch.exp(torch.tensor(logit_scale_value)).item()
            print(f"Computed temperature: {temperature}")
            return torch.tensor(float(logit_scale_value))
        elif hasattr(self.model, 'temperature'):
            # Direct temperature - convert to logit_scale
            temp = self.model.temperature.item() if hasattr(self.model.temperature, 'item') else self.model.temperature
            print(f"Found direct temperature: {temp}")
            logit_scale = torch.log(torch.tensor(1.0 / temp))
            print(f"Converted to logit_scale: {logit_scale.item()}")
            return logit_scale
        elif hasattr(self.model, 't'):
            # Custom format (assuming t = logit_scale)
            t_value = self.model.t.item() if hasattr(self.model.t, 'item') else self.model.t
            print(f"Found t (assuming logit_scale): {t_value}")
            return torch.tensor(float(t_value))
        else:
            # Fallback to standard CLIP initialization (temp = 0.07)
            print("No temperature found, using default temp=0.07 -> logit_scale=2.659")
            logging.warning("Could not detect pretrained temperature, using default 0.07")
            return torch.log(torch.tensor(1.0 / 0.07))

    def _detect_pretrained_temperature(self) -> float:
        """Extract temperature from pretrained CLIP model (kept for backwards compatibility)."""
        logit_scale = self._detect_pretrained_logit_scale()
        return 1.0 / torch.exp(logit_scale).item()
    
    @property
    def temperature(self):
        """Get current temperature with proper clamping (only for standard CLIP)."""
        if self.use_cross_modal_head:
            return None  # LabCLIP doesn't use temperature
        return 1.0 / torch.exp(self.logit_scale)
    
    def set_finetune(self, image_encoder: Optional[bool] = None, text_encoder: Optional[bool] = None):
        """
        Dynamically sets requires_grad for image and/or text encoder.
        """
        if image_encoder is None:
            image_encoder = self.ft_image_encoder
        if text_encoder is None:
            text_encoder = self.ft_text_encoder

        if hasattr(self.model, "visual") and image_encoder is not None:
            for p in self.model.visual.parameters():
                p.requires_grad = image_encoder
            self._ft_image_encoder = image_encoder

        if hasattr(self.model, "transformer") and text_encoder is not None:
            for p in self.model.transformer.parameters():
                p.requires_grad = text_encoder
            self._ft_text_encoder = text_encoder

    def get_alphas(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Returns the learnable alphas from the alignment head (image/text).
        Only applicable for standard CLIP alignment.
        """
        if self.use_cross_modal_head:
            # LabCLIP head may have aggregators with alphas
            if hasattr(self.align_head, 'image_aggregator') and self.align_head.image_aggregator:
                image_alphas = getattr(self.align_head.image_aggregator, 'alphas', None)
            else:
                image_alphas = None
            
            if hasattr(self.align_head, 'text_aggregator') and self.align_head.text_aggregator:
                text_alphas = getattr(self.align_head.text_aggregator, 'alphas', None)
            else:
                text_alphas = None
                
            return {'image': image_alphas, 'text': text_alphas}
        else:
            return self.align_head.get_alphas()

    @torch.no_grad()
    def _maybe_move_to_device(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move tensor to device if needed."""
        if self.assume_inputs_on_device or x is None or x.device == device:
            return x
        return x.to(device, non_blocking=True)

    def extract_features(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
        """
        Extracts multi-layer features for image or text, as dicts.
        Assumes inputs are already on the correct device if `assume_inputs_on_device=True`.
        """
        # Determine the device from the model parameters
        model_device = next(self.model.parameters()).device

        feats = {}
        if image is not None:
            image = self._maybe_move_to_device(image, model_device)
            feats['image'] = extract_intermediate_features(
                image, self.model, device=model_device,
                layer_names=self.image_layer_names, is_image=True, dtype=self.model.dtype
            )
        if text is not None:
            text = self._maybe_move_to_device(text, model_device)
            feats['text'] = extract_intermediate_features(
                text, self.model, device=model_device,
                layer_names=self.text_layer_names, is_image=False, dtype=self.model.dtype
            )
        return feats

    def encode_image(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode image to embeddings."""
        if self.use_cross_modal_head:
            # For LabCLIP FT, we can't return features dict - need embeddings for cache system
            # Process features through the alignment head's image aggregator if available
            feats = self.extract_features(image=image)
            if 'image' in feats and hasattr(self.align_head, 'image_aggregator') and self.align_head.image_aggregator:
                return self.align_head._process_features(
                    feats['image'], 
                    self.align_head.image_layer_names, 
                    self.align_head.image_aggregator
                )
            elif 'image' in feats:
                # Fallback: return final layer features as embeddings
                image_feats = feats['image']
                if isinstance(image_feats, dict):
                    if 'final' in image_feats:
                        return F.normalize(image_feats['final'], dim=-1, eps=1e-6)
                    else:
                        # Return last available layer
                        last_key = next(reversed(image_feats))
                        return F.normalize(image_feats[last_key], dim=-1, eps=1e-6)
                else:
                    return F.normalize(image_feats, dim=-1, eps=1e-6)
            return None
        else:
            feats = self.extract_features(image=image)
            return self.align_head.encode_image(feats.get('image')) if 'image' in feats else None

    def encode_text(self, text: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode text to embeddings."""
        if self.use_cross_modal_head:
            # For LabCLIP FT, we can't return features dict - need embeddings for cache system
            # Process features through the alignment head's text aggregator if available
            feats = self.extract_features(text=text)
            if 'text' in feats and hasattr(self.align_head, 'text_aggregator') and self.align_head.text_aggregator:
                return self.align_head._process_features(
                    feats['text'], 
                    self.align_head.text_layer_names, 
                    self.align_head.text_aggregator
                )
            elif 'text' in feats:
                # Fallback: return final layer features as embeddings
                text_feats = feats['text']
                if isinstance(text_feats, dict):
                    if 'final' in text_feats:
                        return F.normalize(text_feats['final'], dim=-1, eps=1e-6)
                    else:
                        # Return last available layer
                        last_key = next(reversed(text_feats))
                        return F.normalize(text_feats[last_key], dim=-1, eps=1e-6)
                else:
                    return F.normalize(text_feats, dim=-1, eps=1e-6)
            return None
        else:
            feats = self.extract_features(text=text)
            return self.align_head.encode_text(feats.get('text')) if 'text' in feats else None

    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
        """
        Extracts, aligns, and returns (img_embed, text_embed) for standard CLIP
        or scores for LabCLIP, depending on use_cross_modal_head flag.
        """
        if self.use_cross_modal_head:
            # LabCLIP mode: both image and text must be provided
            if image is None or text is None:
                raise ValueError("LabCLIP mode requires both image and text inputs")
            
            # Extract features
            feats = self.extract_features(image=image, text=text)
            
            # Pass through LabCLIP head to get scores
            scores = self.align_head(feats['image'], feats['text'])
            return scores
        else:
            # Standard CLIP mode: return embeddings
            img_embed = self.encode_image(image) if image is not None else None
            txt_embed = self.encode_text(text)  if text  is not None else None
            if img_embed is not None and txt_embed is not None:
                return img_embed, txt_embed
            return img_embed if txt_embed is None else txt_embed

    def to(self, device):
        """Move module to device."""
        super().to(device)
        # Keep submodules on the same device
        self.model = self.model.to(device)
        self.align_head = self.align_head.to(device)
        return self