"""
Flexible CLIP multi-layer alignment module.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Union, Tuple

from .aggregators import MultiLayerCLIPAggregator
from .base_heads import BimodalHead, ScoringHead
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
        """
        Args:
            image_layer_dims: Dimensions for each image layer
            text_layer_dims: Dimensions for each text layer
            aggregator_cfg: Configuration for aggregator
            align_cfg: Configuration for alignment heads
            align_image: Whether to align image features
            align_text: Whether to align text features
            image_layer_names: Names of image layers
            text_layer_names: Names of text layers
            init_temperature: Initial temperature value
            freeze_temperature: Whether to freeze temperature
        """
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
        """Get current temperature with proper clamping."""
        return torch.exp(self.logit_scale)
    
    @property
    def device(self):
        """Cached device property to avoid repeated parameter iteration."""
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def _dict_to_ordered_list(self, features_dict: Dict[str, torch.Tensor], order: List[str]):
        """Convert feature dictionary to ordered list."""
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
        """Encode image features."""
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
        """Encode text features."""
        return self._encode_modality(
            features=text_features,
            layer_names=self.text_layer_names,
            head=self.text_head,
            align_enabled=self.align_text,
        )

    def to(self, device):
        """Move module to device."""
        super().to(device)
        self._device = device
        return self

    def forward(
        self,
        image_features: Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        text_features:  Optional[Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Forward pass through alignment.
        
        Args:
            image_features: Image features (multi-layer or single)
            text_features: Text features (multi-layer or single)
        
        Returns:
            Aligned embeddings (single or tuple)
        """
        img = self.encode_image(image_features) if image_features is not None else None
        txt = self.encode_text(text_features)  if text_features  is not None else None
        if img is not None and txt is not None:
            return img, txt
        return img if txt is None else txt
    


@HeadRegistry.register_scoring_head("cross_modal_mlp")
class FlexibleCLIPCrossModalHead(ScoringHead):
    """
    Modified FlexibleCLIPMultiLayerAlignment for LabCLIP.
    Single head that processes concatenated image-text embeddings.
    Registered as 'cross_modal_mlp' scoring head.
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
        """
        Args:
            image_layer_dims: Dimensions for each image layer
            text_layer_dims: Dimensions for each text layer
            aggregator_cfg: Configuration for aggregator
            align_cfg: Configuration for alignment heads
            mlp_hidden_dims: Hidden dimensions for scorer MLP
            dropout: Dropout rate for MLP
            image_layer_names: Names of image layers
            text_layer_names: Names of text layers
            embed_dim: Output embedding dimension
            use_alignment: Whether to use alignment aggregators
        """
        super().__init__()
        self._device = None
        self.use_alignment = use_alignment
        
        # Create aggregators for image and text
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
        """Build the MLP that maps concatenated embeddings to scalar."""
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
        """Process features through aggregator or return final feature."""
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
        """Move module to device."""
        super().to(device)
        self._device = device
        return self