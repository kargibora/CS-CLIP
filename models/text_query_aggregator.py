"""
Text-Query Aggregator (TQA) Head

Implements the advisor's suggestion:
    "Add a layer pre projection of the CLS token and inject a mini aggregator,
    probably an attention head between the keys of the patches and query of the text embedding."

In CLIP terms:
    v_cls' = v_cls + Attn(q=t, k=v_patches, v=v_patches)
    
Then:
    - Project v_cls' to image space
    - Project text as usual
    - Do InfoNCE/contrastive loss

This creates a caption-conditioned image embedding where the text "selects"
which patches are relevant, making the image representation more compositional.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
import math

from .base_heads import BimodalHead
from .head_registry import HeadRegistry


@HeadRegistry.register_bimodal_head("text_query_aggregator")
class TextQueryAggregatorHead(BimodalHead):
    """
    Text-Query Aggregator Head
    
    Uses cross-attention to create caption-conditioned image embeddings:
        v_cls' = v_cls + scale * Attn(q=t, k=v_patches, v=v_patches)
    
    The text embedding queries the image patches to extract relevant information,
    which is then added to the original CLS token. This enables better
    compositional understanding (attributes, relations, colors).
    
    Key features:
    - Multi-head cross-attention (text queries patches)
    - Residual connection with learnable scale
    - Optional projection heads for image and text
    - Compatible with standard InfoNCE contrastive loss
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        vision_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_residual: bool = True,
        learnable_scale: bool = True,
        initial_scale: float = 0.1,
        use_projection: bool = True,
        projection_dim: Optional[int] = None,
        normalize_output: bool = True,
        **kwargs
    ):
        """
        Args:
            embed_dim: Common embedding dimension for attention
            vision_dim: Native vision transformer dimension (e.g., 768 for ViT-B)
            text_dim: Native text transformer dimension (e.g., 512 for CLIP)
            num_heads: Number of attention heads
            dropout: Dropout probability in attention
            use_residual: Whether to add attention output to v_cls (v_cls + attn)
            learnable_scale: Whether the residual scale is learnable
            initial_scale: Initial value for residual scale
            use_projection: Whether to use projection heads for final embeddings
            projection_dim: Output dimension after projection (defaults to embed_dim)
            normalize_output: Whether to L2-normalize output embeddings
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vision_dim = vision_dim if vision_dim is not None else embed_dim
        self.text_dim = text_dim if text_dim is not None else embed_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.use_projection = use_projection
        self.projection_dim = projection_dim or embed_dim
        self.normalize_output = normalize_output
        
        # Projection layers to common dimension for attention
        if self.vision_dim != embed_dim:
            self.vision_proj = nn.Linear(self.vision_dim, embed_dim, bias=False)
            nn.init.xavier_uniform_(self.vision_proj.weight)
        else:
            self.vision_proj = None
            
        if self.text_dim != embed_dim:
            self.text_proj = nn.Linear(self.text_dim, embed_dim, bias=False)
            nn.init.xavier_uniform_(self.text_proj.weight)
        else:
            self.text_proj = None
        
        # Cross-attention: text queries patches
        # Q = text, K = patches, V = patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # Use batch_first for cleaner code
        )
        
        # Layer norm after attention
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        
        # Learnable scale for residual connection
        if learnable_scale:
            self.residual_scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer("residual_scale", torch.tensor(initial_scale))
        
        # Optional output projection heads
        if use_projection:
            self.image_projection = nn.Linear(embed_dim, self.projection_dim, bias=False)
            self.text_projection = nn.Linear(embed_dim, self.projection_dim, bias=False)
            # Initialize close to identity if dimensions match
            if embed_dim == self.projection_dim:
                nn.init.eye_(self.image_projection.weight)
                nn.init.eye_(self.text_projection.weight)
            else:
                nn.init.xavier_uniform_(self.image_projection.weight)
                nn.init.xavier_uniform_(self.text_projection.weight)
        else:
            self.image_projection = None
            self.text_projection = None
        
        self._log_init_info(learnable_scale)
    
    def _log_init_info(self, learnable_scale):
        """Log initialization information."""
        logging.info("[TextQueryAggregatorHead] Initialized with:")
        logging.info(f"  - Embed dim: {self.embed_dim}")
        logging.info(f"  - Vision dim (native): {self.vision_dim}")
        logging.info(f"  - Text dim (native): {self.text_dim}")
        logging.info(f"  - Num heads: {self.num_heads}")
        logging.info(f"  - Use residual: {self.use_residual}")
        logging.info(f"  - Learnable scale: {learnable_scale}")
        logging.info(f"  - Initial scale: {self.residual_scale.item():.4f}")
        logging.info(f"  - Use projection: {self.use_projection}")
        logging.info(f"  - Projection dim: {self.projection_dim}")
    
    def _project_to_common_space(
        self,
        vision_cls: torch.Tensor,  # [B, D_vision]
        vision_patches: torch.Tensor,  # [B, P, D_vision]
        text_embed: torch.Tensor,  # [B, D_text]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project vision and text features to common embedding space.
        
        Returns:
            vision_cls_proj: [B, embed_dim]
            vision_patches_proj: [B, P, embed_dim]
            text_proj: [B, embed_dim]
        """
        if self.vision_proj is not None:
            vision_cls_proj = self.vision_proj(vision_cls)
            vision_patches_proj = self.vision_proj(vision_patches)
        else:
            vision_cls_proj = vision_cls
            vision_patches_proj = vision_patches
        
        if self.text_proj is not None:
            text_proj = self.text_proj(text_embed)
        else:
            text_proj = text_embed
        
        return vision_cls_proj, vision_patches_proj, text_proj
    
    def aggregate_with_text_query(
        self,
        vision_cls: torch.Tensor,  # [B, D]
        vision_patches: torch.Tensor,  # [B, P, D]
        text_embed: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        """
        Aggregate vision patches using text as query.
        
        v_cls' = v_cls + scale * Attn(q=t, k=v_patches, v=v_patches)
        
        Args:
            vision_cls: CLS token embedding [B, D]
            vision_patches: Patch embeddings [B, P, D]
            text_embed: Text embedding [B, D]
            
        Returns:
            aggregated: Aggregated vision embedding [B, D]
        """
        B = text_embed.shape[0]
        
        # Reshape text for attention: [B, D] -> [B, 1, D] (single query)
        query = text_embed.unsqueeze(1)  # [B, 1, D]
        
        # Keys and values are patches
        key = vision_patches  # [B, P, D]
        value = vision_patches  # [B, P, D]
        
        # Cross-attention: text queries patches
        # Output shape: [B, 1, D]
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key,
            value=value,
            need_weights=True,
        )
        
        # Squeeze back to [B, D]
        attn_output = attn_output.squeeze(1)  # [B, D]
        
        # Apply layer norm
        attn_output = self.attn_layer_norm(attn_output)
        
        # Residual connection with scale
        if self.use_residual:
            aggregated = vision_cls + self.residual_scale * attn_output
        else:
            aggregated = attn_output
        
        return aggregated
    
    def encode_image(
        self,
        image_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        text_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Encode image features with optional text conditioning.
        
        If text_features is provided, creates caption-conditioned embedding.
        Otherwise, just projects the global image features.
        
        Args:
            image_features: Either a tensor [B, D] or dict with 'global' and 'patches' keys
            text_features: Optional text embedding for conditioning [B, D_text]
            
        Returns:
            Image embedding [B, projection_dim]
        """
        # Handle different input formats
        if isinstance(image_features, dict):
            # Prefer unprojected global for TQA since we need consistent dimension with patches
            vision_cls = image_features.get('global_unprojected', 
                         image_features.get('global', 
                         image_features.get('image_global')))
            vision_patches = image_features.get('patches', image_features.get('image_patches'))
            # Debug: log what we received
            if vision_cls is None:
                logging.error(f"[TQA encode_image] Received dict with keys: {image_features.keys()}")
                logging.error(f"[TQA encode_image] 'global': {image_features.get('global')}, 'image_global': {image_features.get('image_global')}")
        else:
            # Assume it's just the global feature
            vision_cls = image_features
            vision_patches = None
        
        if vision_cls is None:
            raise ValueError("Image features must include global/CLS embedding")
        
        # If we have patches and text, do text-query aggregation
        if vision_patches is not None and text_features is not None:
            # Project to common space
            vision_cls_proj, vision_patches_proj, text_proj = self._project_to_common_space(
                vision_cls, vision_patches, text_features
            )
            
            # Aggregate with text query
            aggregated = self.aggregate_with_text_query(
                vision_cls_proj, vision_patches_proj, text_proj
            )
        else:
            # Just project global features
            if self.vision_proj is not None:
                aggregated = self.vision_proj(vision_cls)
            else:
                aggregated = vision_cls
        
        # Apply output projection
        if self.image_projection is not None:
            output = self.image_projection(aggregated)
        else:
            output = aggregated
        
        # Normalize
        if self.normalize_output:
            output = F.normalize(output, dim=-1, eps=1e-6)
        
        return output
    
    def encode_text(
        self,
        text_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """
        Encode text features.
        
        Args:
            text_features: Text embedding [B, D_text] or dict with 'global' key
            
        Returns:
            Text embedding [B, projection_dim]
        """
        # Handle different input formats
        if isinstance(text_features, dict):
            text = text_features.get('global', text_features.get('text_global'))
        else:
            text = text_features
        
        if text is None:
            raise ValueError("Text features must include global embedding")
        
        # Project to common space
        if self.text_proj is not None:
            text_proj = self.text_proj(text)
        else:
            text_proj = text
        
        # Apply output projection
        if self.text_projection is not None:
            output = self.text_projection(text_proj)
        else:
            output = text_proj
        
        # Normalize
        if self.normalize_output:
            output = F.normalize(output, dim=-1, eps=1e-6)
        
        return output
    
    def forward(
        self,
        image_features: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        text_features: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        image_patches: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the TQA head.
        
        Args:
            image_features: Image global features [B, D_vision] or dict
            text_features: Text global features [B, D_text] or dict
            image_patches: Patch features [B, P, D_vision] (optional, can be in image_features dict)
            return_dict: Whether to return a dict or tuple
            
        Returns:
            If return_dict=True:
                Dict with 'image_embeds', 'text_embeds', etc.
                Note: logit_scale is managed by the base model/pipeline, not this head.
            Else:
                Tuple of (image_embeds, text_embeds)
        """
        # Extract patches from dict if provided
        if isinstance(image_features, dict) and image_patches is None:
            image_patches = image_features.get('patches', image_features.get('image_patches'))
            # Prefer unprojected global for consistent dimension with patches
            vision_global = image_features.get('global_unprojected', 
                            image_features.get('global', 
                            image_features.get('image_global')))
        else:
            vision_global = image_features
        
        # Extract text global from dict if needed
        if isinstance(text_features, dict):
            text_global = text_features.get('global', text_features.get('text_global'))
        else:
            text_global = text_features
        
        # Encode both modalities
        # For image, pass text for conditioning if we have patches
        if image_patches is not None and text_global is not None:
            image_input = {'global': vision_global, 'patches': image_patches}
            image_embeds = self.encode_image(image_input, text_features=text_global)
        elif vision_global is not None:
            image_embeds = self.encode_image(vision_global)
        else:
            image_embeds = None
        
        text_embeds = self.encode_text(text_global) if text_global is not None else None
        
        if return_dict:
            output = {
                'image_embeds': image_embeds,
                'text_embeds': text_embeds,
                # Note: logit_scale is now managed by the base model/pipeline, not the head
            }
            
            # Add attention-conditioned embedding if different from base
            if image_patches is not None and text_global is not None:
                # Also compute base image embedding (without conditioning)
                base_image_embeds = self.encode_image({'global': vision_global})
                output['base_image_embeds'] = base_image_embeds
                output['caption_conditioned_image'] = image_embeds
            
            return output
        else:
            return image_embeds, text_embeds


@HeadRegistry.register_bimodal_head("text_query_aggregator_v2")
class TextQueryAggregatorHeadV2(BimodalHead):
    """
    Alternative implementation with more flexibility.
    
    Supports:
    - Multiple attention layers
    - Different attention patterns (text→patches, patches→text, bidirectional)
    - Feed-forward layers after attention
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        vision_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 1,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_mode: str = "text_to_patches",  # "text_to_patches", "patches_to_text", "bidirectional"
        use_residual: bool = True,
        learnable_scale: bool = True,
        initial_scale: float = 0.1,
        normalize_output: bool = True,
        learnable_temperature: bool = False,
        initial_temperature: float = 0.07,
        **kwargs
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vision_dim = vision_dim if vision_dim is not None else embed_dim
        self.text_dim = text_dim if text_dim is not None else embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_mode = attention_mode
        self.use_residual = use_residual
        self.normalize_output = normalize_output
        
        # Projections to common space
        if self.vision_dim != embed_dim:
            self.vision_proj = nn.Linear(self.vision_dim, embed_dim, bias=False)
        else:
            self.vision_proj = None
            
        if self.text_dim != embed_dim:
            self.text_proj = nn.Linear(self.text_dim, embed_dim, bias=False)
        else:
            self.text_proj = None
        
        # Build attention layers
        ffn_dim = ffn_dim or embed_dim * 4
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                    nn.Dropout(dropout),
                )
            )
            self.norm1_layers.append(nn.LayerNorm(embed_dim))
            self.norm2_layers.append(nn.LayerNorm(embed_dim))
        
        # Residual scale
        if learnable_scale:
            self.residual_scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer("residual_scale", torch.tensor(initial_scale))
        
        # Output projections
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.output_proj.weight)
        
        # Temperature
        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * math.log(1.0 / initial_temperature)
            )
        else:
            self.register_buffer(
                "logit_scale",
                torch.ones([]) * math.log(1.0 / initial_temperature)
            )
        
        logging.info(f"[TextQueryAggregatorHeadV2] Initialized with {num_layers} layers, mode={attention_mode}")
    
    @property
    def temperature(self):
        return 1.0 / self.logit_scale.exp()
    
    def aggregate(
        self,
        vision_cls: torch.Tensor,
        vision_patches: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate vision features using cross-attention with text.
        """
        # Project to common space
        if self.vision_proj is not None:
            vision_cls = self.vision_proj(vision_cls)
            vision_patches = self.vision_proj(vision_patches)
        if self.text_proj is not None:
            text_embed = self.text_proj(text_embed)
        
        # Start with text as query
        query = text_embed.unsqueeze(1)  # [B, 1, D]
        kv = vision_patches  # [B, P, D]
        
        # Apply attention layers
        attn_output = query
        for i in range(self.num_layers):
            # Cross-attention
            residual = attn_output
            attn_out, _ = self.attention_layers[i](
                query=attn_output, key=kv, value=kv
            )
            attn_output = self.norm1_layers[i](residual + attn_out)
            
            # FFN
            residual = attn_output
            ffn_out = self.ffn_layers[i](attn_output)
            attn_output = self.norm2_layers[i](residual + ffn_out)
        
        # Squeeze and combine with vision CLS
        attn_output = attn_output.squeeze(1)  # [B, D]
        
        if self.use_residual:
            output = vision_cls + self.residual_scale * attn_output
        else:
            output = attn_output
        
        return self.output_proj(output)
    
    def encode_image(
        self,
        image_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
        text_features: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Encode image with optional text conditioning."""
        if isinstance(image_features, dict):
            # Prefer unprojected global for consistent dimension with patches
            vision_cls = image_features.get('global_unprojected', 
                         image_features.get('global', 
                         image_features.get('image_global')))
            vision_patches = image_features.get('patches', image_features.get('image_patches'))
        else:
            vision_cls = image_features
            vision_patches = None
        
        if vision_patches is not None and text_features is not None:
            output = self.aggregate(vision_cls, vision_patches, text_features)
        else:
            if self.vision_proj is not None:
                output = self.vision_proj(vision_cls)
            else:
                output = vision_cls
            output = self.output_proj(output)
        
        if self.normalize_output:
            output = F.normalize(output, dim=-1, eps=1e-6)
        
        return output
    
    def encode_text(
        self,
        text_features: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Encode text features."""
        if isinstance(text_features, dict):
            text = text_features.get('global', text_features.get('text_global'))
        else:
            text = text_features
        
        if self.text_proj is not None:
            output = self.text_proj(text)
        else:
            output = text
        
        output = self.output_proj(output)
        
        if self.normalize_output:
            output = F.normalize(output, dim=-1, eps=1e-6)
        
        return output
    
    def forward(
        self,
        image_features: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        text_features: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        image_patches: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """Forward pass."""
        # Handle dict inputs
        if isinstance(image_features, dict) and image_patches is None:
            image_patches = image_features.get('patches', image_features.get('image_patches'))
            # Prefer unprojected global for consistent dimension with patches
            vision_global = image_features.get('global_unprojected', 
                            image_features.get('global', 
                            image_features.get('image_global')))
        else:
            vision_global = image_features
        
        if isinstance(text_features, dict):
            text_global = text_features.get('global', text_features.get('text_global'))
        else:
            text_global = text_features
        
        # Encode
        if image_patches is not None and text_global is not None:
            image_input = {'global': vision_global, 'patches': image_patches}
            image_embeds = self.encode_image(image_input, text_features=text_global)
        elif vision_global is not None:
            image_embeds = self.encode_image(vision_global)
        else:
            image_embeds = None
        
        text_embeds = self.encode_text(text_global) if text_global is not None else None
        
        if return_dict:
            return {
                'image_embeds': image_embeds,
                'text_embeds': text_embeds,
                'logit_scale': self.logit_scale.exp(),
            }
        else:
            return image_embeds, text_embeds
