"""
Basic alignment head components for CLIP features.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from .base_heads import AlignmentHead
from .head_registry import HeadRegistry


def build_align_head_by_cfg(in_dim: int, align_cfg: Optional[dict]) -> AlignmentHead:
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
    
    Args:
        in_dim: Input dimension
        align_cfg: Configuration dictionary
    
    Returns:
        Alignment head module (instance of AlignmentHead base class)
    """
    if align_cfg is None:
        # fallback to dense linear head with identity init
        return CLIPFeatureAlignment(in_dim=in_dim)

    cfg = {k: v for k, v in align_cfg.items()}  # shallow copy
    typ = cfg.pop("type", "linear")

    if typ == "linear":
        # use existing CLIPFeatureAlignment
        return CLIPFeatureAlignment(in_dim=in_dim, **cfg)

    elif typ == "lowrank":
        from utils.alignment_heads import LowRankLinearAlignment
        return LowRankLinearAlignment(in_dim=in_dim, **cfg)

    elif typ == "blockdiag":
        from utils.alignment_heads import BlockDiagonalLinearAlignment
        return BlockDiagonalLinearAlignment(in_dim=in_dim, **cfg)

    else:
        raise ValueError(f"Unknown align head type: {typ}")


@HeadRegistry.register_alignment_head("linear")
class CLIPFeatureAlignment(AlignmentHead):
    """
    Generic MLP alignment head for arbitrary CLIP features (intermediate or final).
    Can be used for both image and text representations.
    Registered as 'linear' alignment head.
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
            dtype: Data type for layer initialization.
        """
        super().__init__()
        out_dim = out_dim or in_dim
        self.normalize = normalize
        self.mlp = self._build_mlp(in_dim, out_dim, mlp_hidden_dim, mlp_layers, bias, dtype)

    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dim, num_layers, bias, dtype):
        """Build MLP layers with proper initialization."""
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
        """
        Forward pass through alignment head.
        
        Args:
            features: Input features (batch, in_dim)
        
        Returns:
            Aligned features (batch, out_dim)
        """
        x = self.mlp(features)
        if self.normalize:
            x = F.normalize(x, dim=-1, eps=1e-6)  # safer normalization for fp16/bf16
        return x