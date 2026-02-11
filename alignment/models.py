from __future__ import annotations
import torch
from torch import nn
from typing import Optional, Callable, Dict, List, Tuple, Union
import logging
import copy
from utils.align import extract_intermediate_features

from utils.alignment_heads import BaseAlignmentHead, LowRankLinearAlignment, BlockDiagonalLinearAlignment
import torch.nn.functional as F
from typing import Optional, List, Dict, Union

def _build_align_head_by_cfg(in_dim: int, align_cfg: Optional[dict]) -> nn.Module:
    """
    Factory: build an alignment head from config.
    align_cfg examples:
      {"type": "linear", "out_dim": d, "normalize": True}
      {"type": "lowrank", "out_dim": d, "rank": 64, "normalize": True}
      {"type": "blockdiag",
       "out_dim": d,
       "blocks": [[0,1,2,...],[...],...],
       "out_blocks": [[...],[...],...],  # optional if square
       "mode_per_block": ["dense","diagonal","lowrank"],
       "rank_per_block": [None,None,8],
       "normalize": True}
    """
    if align_cfg is None:
        # fallback to your dense linear head with identity init
        return CLIPFeatureAlignment(in_dim=in_dim)

    cfg = {k: v for k, v in align_cfg.items()}  # shallow copy
    typ = cfg.pop("type", "linear")

    if typ == "linear":
        # use your existing CLIPFeatureAlignment
        return CLIPFeatureAlignment(in_dim=in_dim, **cfg)

    elif typ == "lowrank":
        return LowRankLinearAlignment(in_dim=in_dim, **cfg)

    elif typ == "blockdiag":
        return BlockDiagonalLinearAlignment(in_dim=in_dim, **cfg)

    else:
        raise ValueError(f"Unknown align head type: {typ}")
    
class CLIPFeatureAlignment(nn.Module):
    """
    Generic MLP alignment head for arbitrary CLIP features (intermediate or final).
    Can be used for both image and text representations.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        mlp_hidden_dim: int = 768,
        mlp_layers: int = 1,
        bias: bool = True,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            in_dim: Input feature dimension (from any CLIP layer).
            out_dim: Output embedding dimension. Defaults to in_dim if not set.
            mlp_hidden_dim: Hidden dim for the MLP if mlp_layers > 1.
            mlp_layers: Number of MLP layers (>=1).
            bias: Whether to use bias in non-final layers.
            normalize: If True, output will be L2-normalized.
        """
        super().__init__()
        out_dim = out_dim or in_dim
        self.normalize = normalize
        self.mlp = self._build_mlp(in_dim, out_dim, mlp_hidden_dim, mlp_layers, bias, dtype)

    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dim, num_layers, bias, dtype):
        layers = []
        if num_layers == 1:
            lin = nn.Linear(in_dim, out_dim, bias=False)
            # Identity init if square
            if in_dim == out_dim:
                nn.init.eye_(lin.weight)
            else:
                nn.init.xavier_uniform_(lin.weight)
            layers.append(lin)
        else:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias, dtype=dtype))
                layers.append(nn.ReLU(inplace=True))
            final = nn.Linear(hidden_dim, out_dim, bias=False)
            if hidden_dim == out_dim:
                nn.init.eye_(final.weight)
            else:
                nn.init.xavier_uniform_(final.weight)
            layers.append(final)
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.mlp(features)
        if self.normalize:
            x = F.normalize(x, dim=-1, eps=1e-6)  # safer normalization for fp16/bf16
        return x
    
import logging
from typing import List, Sequence
import torch
from torch import nn

class MultiLayerCLIPAggregator(nn.Module):
    """
    For a list of CLIP features, applies alignment heads and outputs:
        x' = A_h x_h + sum_l alpha_l * A_l x_l
    where x_h is the global token feature (last in features list).
    """
    def __init__(
        self,
        layer_dims: Sequence[int],
        alphas: List[float] = None,
        align_cfg: dict = None,
        learnable_alphas: bool = False,
        init_alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        assume_inputs_on_device: bool = True,
    ):
        """
        Args:
            layer_dims: feature dims per layer; last is the global layer.
            alphas: initial/fixed weights for local layers (length n_layers-1).
            align_cfg: kwargs for CLIPFeatureAlignment (shared across heads).
            learnable_alphas: if True, alpha_l is a learnable parameter; else, buffer.
            init_alpha: used if `alphas` not provided.
            dtype: dtype for alpha parameters/buffers initialization.
            assume_inputs_on_device: skip device moves in forward if True.
        """
        super().__init__()
        layer_dims = list(layer_dims)
        assert len(layer_dims) >= 1, "layer_dims must have at least the global layer"
        self.n_layers = len(layer_dims)
        self.learnable_alphas = learnable_alphas
        self.assume_inputs_on_device = assume_inputs_on_device

        # init alphas (only for local layers, i.e., everything except the last/global)
        if self.n_layers > 1:
            if alphas is not None and len(alphas) > 0:
                if len(alphas) < self.n_layers - 1:
                    raise ValueError(
                        f"If alphas are provided, expected at least {self.n_layers - 1}, got {len(alphas)}."
                    )
                init_alphas = alphas[: self.n_layers - 1]
            else:
                init_alphas = [init_alpha] * (self.n_layers - 1)

            tensor_alphas = torch.tensor(init_alphas, dtype=dtype)
            if learnable_alphas:
                self.alphas = nn.Parameter(tensor_alphas)
            else:
                self.register_buffer("alphas", tensor_alphas, persistent=True)
        else:
            # no local layers; keep a tiny buffer for simpler logic (never used)
            self.register_buffer("alphas", torch.empty(0, dtype=dtype), persistent=True)

        logging.debug(
            f"MultiLayerCLIPAggregator: n_layers={self.n_layers}, "
            f"learnable_alphas={learnable_alphas}, "
            f"assume_inputs_on_device={assume_inputs_on_device}"
        )

        # Build per-layer heads
        cfg = align_cfg or {}
        self.align_heads = nn.ModuleList([
            CLIPFeatureAlignment(in_dim=layer_dims[i], **cfg) for i in range(self.n_layers)
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list/tuple of tensors, length == n_layers
        Returns:
            Aggregated embedding: (batch, dim)
        """
        if not isinstance(features, (list, tuple)):
            raise TypeError("features must be a list/tuple of tensors")
        assert len(features) == self.n_layers, (
            f"Expected {self.n_layers} feature tensors, got {len(features)}."
        )

        # Optionally ensure on-device (skip if caller already guarantees it)
        if not self.assume_inputs_on_device:
            device = next(self.parameters()).device
            features = [
                f if f.device == device else f.to(device, non_blocking=True)
                for f in features
            ]

        # Global (last) layer
        global_feat = self.align_heads[-1](features[-1])
        agg = global_feat

        # Local layers (if any)
        for i in range(self.n_layers - 1):
            local_feat = self.align_heads[i](features[i])
            # make sure alpha matches compute dtype to avoid upcasts in AMP
            alpha = self.alphas[i].to(local_feat.dtype)
            agg = agg + alpha * local_feat

        return agg
    
class FlexibleCLIPMultiLayerAlignment(nn.Module):
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
            # Keep original temperature frozen
            self.register_buffer(
                'logit_scale', 
                torch.log(torch.tensor(1.0 / init_temperature))
            )
        else:
            # Allow temperature to be learned during finetuning
            self.logit_scale = nn.Parameter(
                torch.log(torch.tensor(1.0 / init_temperature))
            )

    @property
    def temperature(self):
        """Get current temperature with proper clamping"""
        return torch.exp(self.logit_scale)
    
    @property
    def device(self):
        """Cached device property to avoid repeated parameter iteration."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def _dict_to_ordered_list(self, features_dict: Dict[str, torch.Tensor], order: List[str]):
        missing = [k for k in order if k not in features_dict]
        if missing:
            raise KeyError(
                f"Missing required feature(s): {missing}. "
                f"Available keys: {list(features_dict.keys())}"
            )
        return [features_dict[k] for k in order]

    def get_alphas(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Returns the learnable alpha tensors (or None) from image/text heads.
        """
        image_alphas = getattr(self.image_head, 'alphas', None)
        text_alphas  = getattr(self.text_head,  'alphas', None)
        return {'image': image_alphas, 'text': text_alphas}

    def _final_feature_from(
        self,
        features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        layer_names: Optional[List[str]],
    ) -> torch.Tensor:
        """
        Resolve the 'final' feature when alignment is disabled or no head is present.
        Preference: dict['final'] -> dict[last named layer] -> last entry -> tensor itself.
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

    def _encode_modality(
        self,
        features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]],
        layer_names: Optional[List[str]],
        head: Optional[nn.Module],
        align_enabled: bool,
    ) -> Optional[torch.Tensor]:
        """
        Shared encoding path for image/text: resolve final when disabled,
        or assemble ordered inputs and run the alignment head.
        Assumes inputs are already on the correct device.
        Normalizes once with eps for fp16/bf16 stability.
        """
        if features is None:
            return None

        if not align_enabled or head is None:
            out = self._final_feature_from(features, layer_names)
            return F.normalize(out, dim=-1, eps=1e-6)

        if isinstance(features, dict):
            assert layer_names is not None, "Layer names required for dict input when alignment is enabled."
            ordered = self._dict_to_ordered_list(features, layer_names)
        else:
            ordered = features  # list or tensor acceptable to head

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
        text_features:  Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        img = self.encode_image(image_features) if image_features is not None else None
        txt = self.encode_text(text_features)  if text_features  is not None else None
        if img is not None and txt is not None:
            return img, txt
        return img if txt is None else txt
    
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
        assume_inputs_on_device: bool = True,   # <- avoids per-call .to()
        init_temperature: Optional[float] = None,
        freeze_temperature: bool = False,
        use_cross_modal_head: bool = False,  # NEW: Use LabCLIP scoring instead of standard alignment
        mlp_hidden_dims: Optional[List[int]] = None,  # NEW: For LabCLIP MLP configuration
        dropout: float = 0.1,  # NEW: For LabCLIP MLP dropout
        embed_dim: Optional[int] = None,  # NEW: For LabCLIP embedding dimension
        use_alignment: bool = True,  # NEW: For LabCLIP aggregator control
    ):
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
            self.align_head = LabCLIPFlexibleAlignment(
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
        """Extract logit_scale directly from pretrained CLIP model"""
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
            # Your custom format (assuming t = logit_scale)
            t_value = self.model.t.item() if hasattr(self.model.t, 'item') else self.model.t
            print(f"Found t (assuming logit_scale): {t_value}")
            return torch.tensor(float(t_value))
        else:
            # Fallback to standard CLIP initialization (temp = 0.07)
            print("No temperature found, using default temp=0.07 -> logit_scale=2.659")
            logging.warning("Could not detect pretrained temperature, using default 0.07")
            return torch.log(torch.tensor(1.0 / 0.07))

    def _detect_pretrained_temperature(self) -> float:
        """Extract temperature from pretrained CLIP model (kept for backwards compatibility)"""
        logit_scale = self._detect_pretrained_logit_scale()
        return 1.0 / torch.exp(logit_scale).item()
    
    @property
    def temperature(self):
        """Get current temperature with proper clamping (only for standard CLIP)"""
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
        super().to(device)
        # keep submodules on the same device
        self.model = self.model.to(device)
        self.align_head = self.align_head.to(device)
        return self
    
class LabCLIPFlexibleAlignment(nn.Module):
    """
    Modified FlexibleCLIPMultiLayerAlignment for LabCLIP.
    Single head that processes concatenated image-text embeddings.
    """
    def __init__(
        self,
        image_layer_dims: Optional[List[int]] = None,
        text_layer_dims: Optional[List[int]] = None,
        aggregator_cfg: Optional[dict] = None,
        align_cfg: Optional[dict] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        image_layer_names: Optional[List[str]] = None,
        text_layer_names: Optional[List[str]] = None,
        embed_dim: Optional[int] = None,
        use_alignment: bool = True,
    ):
        super().__init__()
        self._device = None
        self.use_alignment = use_alignment
        
        # Create aggregators for image and text (same as original)
        self.image_layer_names = image_layer_names or [f"layer_{i}" for i in range(len(image_layer_dims))] if image_layer_dims else None
        self.text_layer_names = text_layer_names or [f"layer_{i}" for i in range(len(text_layer_dims))] if text_layer_dims else None
        
        # Image aggregator - only create if alignment is enabled
        if use_alignment and image_layer_dims is not None:
            self.image_aggregator = MultiLayerCLIPAggregator(
                layer_dims=image_layer_dims,
                align_cfg=align_cfg,
                **(aggregator_cfg or {})
            )
        else:
            self.image_aggregator = None
            
        # Text aggregator - only create if alignment is enabled  
        if use_alignment and text_layer_dims is not None:
            self.text_aggregator = MultiLayerCLIPAggregator(
                layer_dims=text_layer_dims,
                align_cfg=align_cfg,
                **(aggregator_cfg or {})
            )
        else:
            self.text_aggregator = None
        
        # Determine output embedding dimension
        if embed_dim is not None:
            final_embed_dim = embed_dim
        elif align_cfg and 'out_dim' in align_cfg:
            final_embed_dim = align_cfg['out_dim']
        elif image_layer_dims:
            final_embed_dim = image_layer_dims[-1]
        elif text_layer_dims:
            final_embed_dim = text_layer_dims[-1]
        else:
            final_embed_dim = 512  # reasonable default
            
        # Single MLP head for concatenated embeddings -> scalar
        self.mlp_head = self._build_scorer_mlp(
            embed_dim=final_embed_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout
        )
        
        logging.info(
            f"Initialized LabCLIPFlexibleAlignment "
            f"(use_alignment={use_alignment}, embed_dim={final_embed_dim}, "
            f"image layers: {self.image_layer_names}, text layers: {self.text_layer_names})"
        )
    
    def _build_scorer_mlp(self, embed_dim: int, hidden_dims: Optional[List[int]], dropout: float) -> nn.Module:
        """Build the MLP that maps concatenated embeddings to scalar"""
        layers = []
        input_dim = 2 * embed_dim  # Concatenated [image, text]
        
        if hidden_dims is None or len(hidden_dims) == 0:
            # Simple linear layer
            layers.append(nn.Linear(input_dim, 1))
        else:
            # Multi-layer MLP
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            
            # Final layer to scalar
            layers.append(nn.Linear(input_dim, 1))
        
        # Add tanh to ensure output in [-1, 1]
        layers.append(nn.Tanh())
        
        mlp = nn.Sequential(*layers)
        
        # Initialize weights
        for module in mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        return mlp
    
    @property
    def device(self):
        """Cached device property."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    
    def _process_features(
        self,
        features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        layer_names: Optional[List[str]],
        aggregator: Optional[nn.Module],
    ) -> torch.Tensor:
        """Process features through aggregator or return final feature"""
        if aggregator is None or not self.use_alignment:
            # No aggregator or alignment disabled - use raw embeddings
            if isinstance(features, dict):
                if "final" in features:
                    feat = features["final"]
                elif layer_names and layer_names[-1] in features:
                    feat = features[layer_names[-1]]
                else:
                    last_key = next(reversed(features))
                    feat = features[last_key]
            elif isinstance(features, list):
                feat = features[-1]
            else:
                feat = features  # already a tensor
            
            return F.normalize(feat, dim=-1, eps=1e-6)
        
        # Use aggregator for alignment
        if isinstance(features, dict):
            assert layer_names is not None, "Layer names required for dict input"
            ordered = [features[k] for k in layer_names]
        else:
            ordered = features
        
        return aggregator(ordered)
    
    def forward(
        self,
        image_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        text_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass: aggregate features, concatenate, and score.
        
        Args:
            image_features: Image features (multi-layer or single)
            text_features: Text features (multi-layer or single)
        
        Returns:
            scores: (batch_size,) tensor with values in [-1, 1]
        """
        # Process image features
        image_embed = self._process_features(
            image_features, 
            self.image_layer_names,
            self.image_aggregator
        )
        
        # Process text features
        text_embed = self._process_features(
            text_features,
            self.text_layer_names,
            self.text_aggregator
        )
        
        # Concatenate embeddings
        combined = torch.cat([image_embed, text_embed], dim=-1)
        
        # Pass through MLP head to get scalar
        scores = self.mlp_head(combined).squeeze(-1)
        
        return scores
    
    def to(self, device):
        super().to(device)
        self._device = device
        return self