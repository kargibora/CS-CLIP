"""
Fine-tunable CLIP wrapper with multi-layer alignment support.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Union, Tuple

from utils.align import extract_intermediate_features
from .aggregators import MultiLayerCLIPAggregator
from .base_heads import BimodalHead
from .head_registry import HeadRegistry


@HeadRegistry.register_bimodal_head("flexible_multi_layer")
class FlexibleCLIPMultiLayerAlignment(BimodalHead):
    """
    High-level alignment wrapper for CLIP.
    Uses MultiLayerCLIPAggregator if intermediate features are provided (for image/text).
    Otherwise falls back to identity on the final-layer feature (with safe L2 norm).
    Assumes inputs are already on the correct device (no per-call device transfers).
    """
    def __init__(
        self,
        image_layer_dims: Optional[List[int]] = None,
        text_layer_dims: Optional[List[int]] = None,
        aggregator_cfg: Optional[dict] = None,
        align_cfg: Optional[dict] = None,
        align_image: bool = False,
        align_text: bool = True,
        image_layer_names: Optional[List[str]] = None,
        text_layer_names: Optional[List[str]] = None,
        init_temperature: float = 1,
        freeze_temperature: bool = False,
    ):
        super().__init__()
        self.align_image = align_image
        self.align_text = align_text
        self._device = None

        # Image head
        if align_image and image_layer_dims is not None:
            self.image_layer_names = image_layer_names or [f"layer_{i}" for i in range(len(image_layer_dims))]
            self.image_head = MultiLayerCLIPAggregator(
                layer_dims=image_layer_dims,
                align_cfg=align_cfg,
                **(aggregator_cfg or {})
            )
        else:
            self.image_layer_names = None
            self.image_head = None

        # Text head
        if align_text and text_layer_dims is not None:
            self.text_layer_names = text_layer_names or [f"layer_{i}" for i in range(len(text_layer_dims))]
            self.text_head = MultiLayerCLIPAggregator(
                layer_dims=text_layer_dims,
                align_cfg=align_cfg,
                **(aggregator_cfg or {})
            )
        else:
            self.text_layer_names = None
            self.text_head = None

        logging.info(
            f"Initialized FlexibleCLIPMultiLayerAlignment "
            f"(image layers: {self.image_layer_names}, text layers: {self.text_layer_names})"
        )

        if freeze_temperature:
            self.register_buffer(
                'logit_scale', 
                torch.log(torch.tensor(1.0 / init_temperature))
            )
        else:
            self.logit_scale = nn.Parameter(
                torch.log(torch.tensor(1.0 / init_temperature))
            )

    @property
    def temperature(self):
        return torch.exp(self.logit_scale)
    
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def _dict_to_ordered_list(self, features_dict: Dict[str, torch.Tensor], order: List[str]):
        missing = [k for k in order if k not in features_dict]
        if missing:
            raise KeyError(f"Missing required feature(s): {missing}. Available keys: {list(features_dict.keys())}")
        return [features_dict[k] for k in order]

    def get_alphas(self) -> Dict[str, Optional[torch.Tensor]]:
        image_alphas = getattr(self.image_head, 'alphas', None)
        text_alphas = getattr(self.text_head, 'alphas', None)
        return {'image': image_alphas, 'text': text_alphas}

    def _final_feature_from(
        self,
        features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        layer_names: Optional[List[str]],
    ) -> torch.Tensor:
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
            return features

    def _encode_modality(
        self,
        features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]],
        layer_names: Optional[List[str]],
        head: Optional[nn.Module],
        align_enabled: bool,
    ) -> Optional[torch.Tensor]:
        if features is None:
            return None

        if not align_enabled or head is None:
            out = self._final_feature_from(features, layer_names)
            return F.normalize(out, dim=-1, eps=1e-6)

        if isinstance(features, dict):
            assert layer_names is not None, "Layer names required for dict input when alignment is enabled."
            ordered = self._dict_to_ordered_list(features, layer_names)
        else:
            ordered = features

        out = head(ordered)
        return out

    def encode_image(
        self,
        image_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        return self._encode_modality(
            features=image_features,
            layer_names=self.image_layer_names,
            head=self.image_head,
            align_enabled=self.align_image,
        )

    def encode_text(
        self,
        text_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        return self._encode_modality(
            features=text_features,
            layer_names=self.text_layer_names,
            head=self.text_head,
            align_enabled=self.align_text,
        )

    def to(self, device):
        super().to(device)
        self._device = device
        return self

    def forward(
        self,
        image_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        text_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        img = self.encode_image(image_features) if image_features is not None else None
        txt = self.encode_text(text_features) if text_features is not None else None
        if img is not None and txt is not None:
            return img, txt
        return img if txt is None else txt


class CLIPMultiLayerFTAlignment(nn.Module):
    """
    Fine-tunable CLIP wrapper that aligns multi-layer features via a FlexibleCLIPMultiLayerAlignment head.
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
        """
        super().__init__()
        self.model = model
        self.align_image = align_image
        self.align_text = align_text
        self.assume_inputs_on_device = assume_inputs_on_device

        self.ft_image_encoder = ft_image_encoder
        self.ft_text_encoder = ft_text_encoder
        self.ft_start_epoch = ft_start_epoch

        # Set (un)freezing according to flags
        self.set_finetune(image_encoder=ft_image_encoder, text_encoder=ft_text_encoder)

        logging.info(
            "CLIPMultiLayerFTAlignment: ft_image_encoder=%s, ft_text_encoder=%s, "
            "align_image=%s, align_text=%s",
            ft_image_encoder, ft_text_encoder, align_image, align_text
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

        # Flexible alignment head
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
        self.text_layer_names = text_layer_names

        # Temperature initialization
        if init_temperature is None:
            init_logit_scale = self._detect_pretrained_logit_scale()
        else:
            init_logit_scale = torch.log(torch.tensor(1.0 / init_temperature))
        
        if freeze_temperature:
            self.register_buffer('logit_scale', init_logit_scale)
        else:
            self.logit_scale = nn.Parameter(init_logit_scale)

    def _detect_pretrained_logit_scale(self) -> torch.Tensor:
        """Extract logit_scale directly from pretrained CLIP model."""
        if hasattr(self.model, 'logit_scale'):
            logit_scale_value = self.model.logit_scale.item() if hasattr(self.model.logit_scale, 'item') else self.model.logit_scale
            return torch.tensor(float(logit_scale_value))
        elif hasattr(self.model, 'temperature'):
            temp = self.model.temperature.item() if hasattr(self.model.temperature, 'item') else self.model.temperature
            return torch.log(torch.tensor(1.0 / temp))
        elif hasattr(self.model, 't'):
            t_value = self.model.t.item() if hasattr(self.model.t, 'item') else self.model.t
            return torch.tensor(float(t_value))
        else:
            logging.warning("Could not detect pretrained temperature, using default 0.07")
            return torch.log(torch.tensor(1.0 / 0.07))

    def _detect_pretrained_temperature(self) -> float:
        """Extract temperature from pretrained CLIP model."""
        logit_scale = self._detect_pretrained_logit_scale()
        return 1.0 / torch.exp(logit_scale).item()
    
    @property
    def temperature(self):
        """Get current temperature."""
        return 1.0 / torch.exp(self.logit_scale)
    
    def set_finetune(self, image_encoder: Optional[bool] = None, text_encoder: Optional[bool] = None):
        """Dynamically sets requires_grad for image and/or text encoder."""
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
        """Returns the learnable alphas from the alignment head."""
        return self.align_head.get_alphas()

    @torch.no_grad()
    def _maybe_move_to_device(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move tensor to device if needed."""
        if self.assume_inputs_on_device or x is None or x.device == device:
            return x
        return x.to(device, non_blocking=True)

    def extract_features(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
        """Extracts multi-layer features for image or text, as dicts."""
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
        feats = self.extract_features(image=image)
        return self.align_head.encode_image(feats.get('image')) if 'image' in feats else None

    def encode_text(self, text: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode text to embeddings."""
        feats = self.extract_features(text=text)
        return self.align_head.encode_text(feats.get('text')) if 'text' in feats else None

    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
        """Extracts, aligns, and returns (img_embed, text_embed)."""
        img_embed = self.encode_image(image) if image is not None else None
        txt_embed = self.encode_text(text) if text is not None else None
        if img_embed is not None and txt_embed is not None:
            return img_embed, txt_embed
        return img_embed if txt_embed is None else txt_embed

    def to(self, device):
        """Move module to device."""
        super().to(device)
        self.model = self.model.to(device)
        self.align_head = self.align_head.to(device)
        return self