"""
Token-Conditioned Alignment (TCA) Head

This head implements token-patch grouping to create caption-conditioned image embeddin        print(f"  - Learnable temperature: {learnable_temperature}")
        print(f"  - Initial temperature: {initial_temperature}")
        print(f"  - Initial logit_scale (buffer): {self.logit_scale.item():.4f} (exp={self.logit_scale.exp().item():.4f})")
It requires access to patch-level (image) and to        # Build output dictionary
        output = {
            # For global CLIP loss: use ORIGINAL CLIP embeddings (not aligned!)
            # This ensures good initial accuracy from pretrained CLIP
            'image_embeds': image_features,  # Original CLIP image embedding
            'text_embeds': text_features,    # Original CLIP text embedding
            
            # For TCA-specific losses: use aligned/conditioned embeddings
            'image_aligned': image_aligned,  # A(img_global) - for pair-aware loss if needed
            'text_aligned': text_aligned,    # B(txt_global) - for pair-aware loss if needed
            'caption_conditioned_image': caption_conditioned_image,
            'token_grouped_patches': token_grouped_patches,
            'text_tokens': text_tokens_normalized,  # Return PROJECTED+NORMALIZED tokens for loss computation
            'logit_scale': self.logit_scale.exp()
        }
        
        # Debug logging (only first time)
        if not hasattr(self, '_debug_logged'):
            import logging
            logging.info(f"[TCA Head Output] logit_scale: {self.logit_scale.exp().item():.4f}")
            logging.info(f"[TCA Head Output] image_features (raw CLIP) norm: {torch.norm(image_features[0]).item():.4f}")
            logging.info(f"[TCA Head Output] text_features (raw CLIP) norm: {torch.norm(text_features[0]).item():.4f}")
            logging.info(f"[TCA Head Output] Alignment used - text: {self.use_text_alignment}, image: {self.use_image_alignment}")
            self._debug_logged = True
        
        # Add intermediate outputs if requestedfeatures, not just global embeddings.

Key components:
1. Token-patch similarity computation
2. Sparse weighted aggregation of patches per token
3. Pooling to create global caption-conditioned image embedding
4. Support for pair-aware contrastive learning with hard negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .base_heads import BaseHead
from .head_registry import HeadRegistry


@HeadRegistry.register_alignment_head("token_conditioned")
class TokenConditionedHead(BaseHead):
    """
    Token-Conditioned Alignment Head
    
    Instead of using only global [CLS] embeddings, this head:
    1. Computes token-patch similarities
    2. Groups patches based on each token's attention
    3. Creates caption-conditioned image representations
    4. Supports local token-level alignment
    
    This enables better compositional understanding (attributes, relations, colors).
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        vision_dim: Optional[int] = None,  # Native vision transformer dimension (e.g., 768)
        text_dim: Optional[int] = None,    # Native text transformer dimension (e.g., 512)
        num_layers: int = 1,
        use_text_alignment: bool = True,
        use_image_alignment: bool = True,
        learnable_temperature: bool = True,
        initial_temperature: float = 0.07,
        sparsify_threshold: Optional[float] = None,  # e.g., 1/num_patches
        pooling_method: str = "mean",  # "mean", "max", "attention"
        normalize_similarity: bool = True,
        min_max_normalize: bool = True,
        **kwargs
    ):
        """
        Args:
            embed_dim: Output embedding dimension (for global features)
            vision_dim: Native vision transformer dimension (if different from embed_dim)
            text_dim: Native text transformer dimension (if different from embed_dim)
            num_layers: Number of alignment layers (currently supports 1)
            use_text_alignment: Whether to use text alignment matrix A
            use_image_alignment: Whether to use image alignment matrix B
            learnable_temperature: Whether temperature is learnable
            initial_temperature: Initial temperature value
            sparsify_threshold: Threshold for sparsifying token-patch weights (None = no sparsification)
            pooling_method: How to pool token-grouped patches to global ("mean", "max", "attention")
            normalize_similarity: Whether to normalize token-patch similarities
            min_max_normalize: Whether to min-max normalize before sparsification
        """
        super().__init__()
        
        # Debug: Log what we receive
        import logging
        logging.info(f"[TokenConditionedHead.__init__] Received - embed_dim: {embed_dim}, vision_dim: {vision_dim}, text_dim: {text_dim}")
        logging.info(f"[TokenConditionedHead.__init__] kwargs: {kwargs}")
        
        self.embed_dim = embed_dim
        self.vision_dim = vision_dim if vision_dim is not None else embed_dim
        self.text_dim = text_dim if text_dim is not None else embed_dim
        self.num_layers = num_layers
        self.use_text_alignment = use_text_alignment
        self.use_image_alignment = use_image_alignment
        self.sparsify_threshold = sparsify_threshold
        self.pooling_method = pooling_method
        self.normalize_similarity = normalize_similarity
        self.min_max_normalize = min_max_normalize
        
        # Projections for patches and tokens to common similarity space
        # These are separate from the global alignment matrices
        if self.vision_dim != self.embed_dim:
            self.patch_projection = nn.Linear(self.vision_dim, self.embed_dim, bias=False)
        else:
            self.patch_projection = None
            
        if self.text_dim != self.embed_dim:
            self.token_projection = nn.Linear(self.text_dim, self.embed_dim, bias=False)
        else:
            self.token_projection = None
        
        # Alignment matrices A (text) and B (image) for GLOBAL features
        # Initialize as identity when input_dim == output_dim
        if use_text_alignment:
            self.text_alignment = nn.Linear(embed_dim, embed_dim, bias=False)
            # Initialize as identity (starts neutral, learns alignment from there)
            nn.init.eye_(self.text_alignment.weight)
        else:
            self.text_alignment = None
            
        if use_image_alignment:
            self.image_alignment = nn.Linear(embed_dim, embed_dim, bias=False)
            # Initialize as identity (starts neutral, learns alignment from there)
            nn.init.eye_(self.image_alignment.weight)
        else:
            self.image_alignment = None
        
        # Temperature for contrastive loss
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / initial_temperature)))
            print(f"  ⚠️  WARNING: Temperature is LEARNABLE - may drift during training!")
        else:
            self.register_buffer("logit_scale", torch.ones([]) * torch.log(torch.tensor(1.0 / initial_temperature)))
            print(f"  ✅ Temperature is FROZEN at {initial_temperature} (buffer, not parameter)")
        
        # Optional: learnable pooling weights for attention-based pooling
        if pooling_method == "attention":
            self.pooling_attention = nn.Linear(embed_dim, 1)
        
        print("[TokenConditionedHead] Initialized with:")
        print(f"  - Embed dim (output): {embed_dim}")
        print(f"  - Vision dim (native): {self.vision_dim}")
        print(f"  - Text dim (native): {self.text_dim}")
        print(f"  - Patch projection: {'Yes' if self.patch_projection is not None else 'No (same as embed_dim)'}")
        print(f"  - Token projection: {'Yes' if self.token_projection is not None else 'No (same as embed_dim)'}")
        print(f"  - Sparsify threshold: {sparsify_threshold if sparsify_threshold is not None else '1/P (auto)'}")
        print(f"  - Pooling method: {pooling_method}")
        print(f"  - Min-max normalize: {min_max_normalize}")
        print(f"  - Use text alignment: {use_text_alignment}")
        print(f"  - Use image alignment: {use_image_alignment}")
        print(f"  - Learnable temperature: {learnable_temperature}")
        print(f"  - Initial temperature: {initial_temperature}")
        print(f"  - Initial logit_scale (buffer): {self.logit_scale.item():.4f} (exp={self.logit_scale.exp().item():.4f})")
    
    def compute_token_patch_similarity(
        self,
        text_tokens: torch.Tensor,  # [B, L, D_text]
        image_patches: torch.Tensor,  # [B, P, D_vision]
    ) -> torch.Tensor:
        """
        Compute similarity matrix between text tokens and image patches.
        
        Args:
            text_tokens: Token-level text features [B, L, D_text]
            image_patches: Patch-level image features [B, P, D_vision]
            
        Returns:
            similarity: [B, L, P] similarity scores
        """
        # Project to common space if needed
        if self.token_projection is not None:
            text_tokens = self.token_projection(text_tokens)
        if self.patch_projection is not None:
            image_patches = self.patch_projection(image_patches)
        
        # Normalize for cosine similarity
        text_tokens = F.normalize(text_tokens, dim=-1)
        image_patches = F.normalize(image_patches, dim=-1)
        
        # Compute similarity: [B, L, D] @ [B, D, P] -> [B, L, P]
        similarity = torch.bmm(text_tokens, image_patches.transpose(1, 2))
        
        return similarity
    
    def sparsify_and_normalize_weights(
        self,
        similarity: torch.Tensor,  # [B, L, P]
    ) -> torch.Tensor:
        """
        Sparsify and normalize token-patch weights.
        
        For each token:
        1. Optionally min-max normalize across patches
        2. Sparsify (set values below threshold to 0)
        3. L1-normalize to create valid weights
        
        Args:
            similarity: [B, L, P] raw similarity scores
            
        Returns:
            weights: [B, L, P] normalized weights (sum to 1 per token)
        """
        weights = similarity
        
        # Min-max normalization per token (optional)
        if self.min_max_normalize:
            # Per token: (s - min) / (max - min)
            min_vals = weights.min(dim=-1, keepdim=True)[0]
            max_vals = weights.max(dim=-1, keepdim=True)[0]
            weights = (weights - min_vals) / (max_vals - min_vals + 1e-8)
        
        # Sparsify: set values below threshold to 0
        # Default threshold: 1/P (paper suggestion) if not explicitly set
        if self.sparsify_threshold is not None:
            threshold = self.sparsify_threshold
        else:
            # Automatically compute as 1/P where P is number of patches
            num_patches = similarity.shape[-1]
            threshold = 1.0 / num_patches
        
        weights = torch.where(
            weights >= threshold,
            weights,
            torch.zeros_like(weights)
        )
        
        # L1 normalize per token (each token's weights sum to 1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights
    
    def aggregate_patches_per_token(
        self,
        weights: torch.Tensor,  # [B, L, P]
        image_patches: torch.Tensor,  # [B, P, D]
    ) -> torch.Tensor:
        """
        Aggregate patches for each token based on weights.
        
        Creates token-conditioned patch representations c_ell.
        
        Args:
            weights: [B, L, P] normalized attention weights
            image_patches: [B, P, D] patch features
            
        Returns:
            token_grouped_patches: [B, L, D] aggregated patch vectors per token
        """
        # [B, L, P] @ [B, P, D] -> [B, L, D]
        token_grouped_patches = torch.bmm(weights, image_patches)
        
        return token_grouped_patches
    
    def pool_to_global(
        self,
        token_grouped_patches: torch.Tensor,  # [B, L, D]
        text_tokens: Optional[torch.Tensor] = None,  # [B, L, D] for attention pooling (projected & normalized)
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] mask for valid tokens
    ) -> torch.Tensor:
        """
        Pool token-grouped patches to a single global caption-conditioned image embedding.
        
        Args:
            token_grouped_patches: [B, L, D] per-token patch aggregations
            text_tokens: [B, L, D] projected & normalized text tokens (for attention pooling)
            attention_mask: [B, L] mask for valid tokens (1=valid, 0=padding)
            
        Returns:
            global_img_text: [B, D] caption-conditioned global image embedding
        """
        if self.pooling_method == "mean":
            # Mean pooling over valid tokens only
            if attention_mask is not None:
                # Sum over valid tokens and divide by count
                mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
                sum_pooled = (token_grouped_patches * mask_expanded).sum(dim=1)  # [B, D]
                count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1], avoid div by zero
                return sum_pooled / count
            else:
                return token_grouped_patches.mean(dim=1)
        
        elif self.pooling_method == "max":
            # Max pooling across tokens
            if attention_mask is not None:
                # Set padding positions to -inf before max
                mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
                masked_patches = token_grouped_patches.masked_fill(mask_expanded == 0, float('-inf'))
                return masked_patches.max(dim=1)[0]
            else:
                return token_grouped_patches.max(dim=1)[0]
        
        elif self.pooling_method == "attention":
            # Attention-weighted pooling
            if text_tokens is None:
                raise ValueError("text_tokens required for attention pooling")
            
            # Compute attention scores from projected & normalized tokens
            scores = self.pooling_attention(text_tokens).squeeze(-1)  # [B, L]
            
            # Mask out padding tokens
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
            # Softmax to get weights
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
            
            # Weighted sum
            return (weights * token_grouped_patches).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_patches: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] - 1 for real tokens, 0 for padding
        return_alignment_matrices: bool = False,
        return_intermediate: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for Token-Conditioned Alignment.
        
        Args:
            image_features: [B, D] global image features (e.g., [CLS] token)
            text_features: [B, D] global text features (e.g., [CLS] token)
            image_patches: [B, P, D] patch-level image features (REQUIRED)
            text_tokens: [B, L, D] token-level text features (REQUIRED)
            attention_mask: [B, L] mask for valid tokens (1=valid, 0=padding)
            return_alignment_matrices: Whether to return A, B matrices
            return_intermediate: Whether to return intermediate computations
            
        Returns:
            Dictionary with:
                - image_embeds: [B, D] aligned global image features
                - text_embeds: [B, D] aligned global text features
                - caption_conditioned_image: [B, D] caption-conditioned global image
                - token_grouped_patches: [B, L, D] per-token patch aggregations (in projected space)
                - text_tokens: [B, L, D] projected+normalized text tokens (for local loss alignment)
                - token_patch_weights: [B, L, P] attention weights (if return_intermediate)
                - logit_scale: temperature parameter
        """
        if image_patches is None or text_tokens is None:
            raise ValueError(
                "TokenConditionedHead requires patch and token features. "
                "Set image_patches and text_tokens in forward() call."
            )
        
        # 1. Apply alignment to global features (standard CLIP alignment)
        if self.use_image_alignment and self.image_alignment is not None:
            image_aligned = self.image_alignment(image_features)
        else:
            image_aligned = image_features
            
        if self.use_text_alignment and self.text_alignment is not None:
            text_aligned = self.text_alignment(text_features)
        else:
            text_aligned = text_features
        
        # 2. Project tokens and patches to common space (and keep for later use)
        # These will be used for both similarity computation AND aggregation
        if self.token_projection is not None:
            text_tokens_projected = self.token_projection(text_tokens)
        else:
            text_tokens_projected = text_tokens
            
        if self.patch_projection is not None:
            image_patches_projected = self.patch_projection(image_patches)
        else:
            image_patches_projected = image_patches
        
        # Normalize for similarity computation
        text_tokens_normalized = F.normalize(text_tokens_projected, dim=-1)
        image_patches_normalized = F.normalize(image_patches_projected, dim=-1)
        
        # 3. Compute token-patch similarities using normalized features
        # Detach to avoid early instability (optional, can be removed later)
        similarity = torch.bmm(
            text_tokens_normalized.detach(),
            image_patches_normalized.detach().transpose(1, 2)
        )  # [B, L, P]
        
        # Debug: Check for NaN in similarity
        if torch.isnan(similarity).any():
            import logging
            logging.error(f"NaN in similarity matrix! Shape: {similarity.shape}, NaN count: {torch.isnan(similarity).sum().item()}")
            logging.error(f"Text tokens - min: {text_tokens.min().item():.6f}, max: {text_tokens.max().item():.6f}, mean: {text_tokens.mean().item():.6f}")
            logging.error(f"Image patches - min: {image_patches.min().item():.6f}, max: {image_patches.max().item():.6f}, mean: {image_patches.mean().item():.6f}")
        
        # 4. Sparsify and normalize to get attention weights
        # DON'T mask before min-max - it causes NaN! Mask AFTER weight computation.
        weights = self.sparsify_and_normalize_weights(similarity)  # [B, L, P]
        
        # 5. Apply attention mask to zero out padding token rows AFTER weight computation
        if attention_mask is not None:
            # Zero out padded token rows safely
            weights = weights * attention_mask.unsqueeze(-1)  # [B, L, 1] broadcast to [B, L, P]
            
            # Check for empty rows (all zeros after masking)
            row_sums = weights.sum(dim=-1, keepdim=True)  # [B, L, 1]
            empty = (row_sums <= 1e-12)
            
            # Fallback to uniform weights over patches for empty rows
            if empty.any():
                B, L, P = weights.shape
                uniform = torch.full_like(weights, 1.0 / P)
                weights = torch.where(empty, uniform, weights)
            
            # Renormalize to ensure sum=1 per token
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Debug: Check for NaN in weights
        if torch.isnan(weights).any():
            import logging
            logging.error(f"NaN in attention weights! Shape: {weights.shape}, NaN count: {torch.isnan(weights).sum().item()}")
        
        # 6. Aggregate patches per token using NORMALIZED projected patches
        # This ensures the aggregation happens in the same space as similarity computation
        token_grouped_patches = self.aggregate_patches_per_token(
            weights,
            image_patches_normalized  # Use normalized patches for consistency
        )  # [B, L, D]
        
        # Apply attention mask to zero out padding token representations
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
            token_grouped_patches = token_grouped_patches * mask_expanded
        
        # Debug: Check for NaN in aggregation
        if torch.isnan(token_grouped_patches).any():
            import logging
            logging.error(f"NaN in token_grouped_patches! Shape: {token_grouped_patches.shape}, NaN count: {torch.isnan(token_grouped_patches).sum().item()}")
        
        # 7. Pool to global caption-conditioned image embedding
        # Use projected tokens for attention pooling (same space as similarity)
        caption_conditioned_image = self.pool_to_global(
            token_grouped_patches,
            text_tokens_normalized if self.pooling_method == "attention" else None,
            attention_mask=attention_mask
        )  # [B, D]
        
        # Debug: Check for NaN in final output
        if torch.isnan(caption_conditioned_image).any():
            import logging
            logging.error(f"NaN in caption_conditioned_image! Shape: {caption_conditioned_image.shape}, NaN count: {torch.isnan(caption_conditioned_image).sum().item()}")
        
        # Build output dictionary
        output = {
            # For global CLIP loss: use ORIGINAL CLIP embeddings (not aligned!)
            # This ensures good initial accuracy from pretrained CLIP
            'image_embeds': image_features,  # Original CLIP image embedding
            'text_embeds': text_features,    # Original CLIP text embedding
            
            # For TCA-specific losses: use aligned/conditioned embeddings
            'image_aligned': image_aligned,  # A(img_global) - for pair-aware loss if needed
            'text_aligned': text_aligned,    # B(txt_global) - for pair-aware loss if needed
            'caption_conditioned_image': caption_conditioned_image,
            'token_grouped_patches': token_grouped_patches,
            'text_tokens': text_tokens_normalized,  # Return PROJECTED+NORMALIZED tokens for loss computation
            'logit_scale': self.logit_scale.exp()
        }
        
        if return_intermediate:
            output['token_patch_weights'] = weights
            output['token_patch_similarity'] = similarity
        
        if return_alignment_matrices:
            output['text_alignment'] = self.text_alignment.weight if self.text_alignment is not None else None
            output['image_alignment'] = self.image_alignment.weight if self.image_alignment is not None else None
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving/loading."""
        config = super().get_config()
        config.update({
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'sparsify_threshold': self.sparsify_threshold,
            'pooling_method': self.pooling_method,
            'normalize_similarity': self.normalize_similarity,
            'min_max_normalize': self.min_max_normalize,
        })
        return config
