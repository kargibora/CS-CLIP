"""
Multi-layer feature aggregators for CLIP.
"""

from __future__ import annotations
import torch
from torch import nn
import logging
from typing import List, Sequence

from .alignment_heads import CLIPFeatureAlignment
from .base_heads import AlignmentHead
from .head_registry import HeadRegistry


@HeadRegistry.register_alignment_head("multi_layer_aggregator")
class MultiLayerCLIPAggregator(AlignmentHead):
    """
    For a list of CLIP features, applies alignment heads and outputs:
        x' = A_h x_h + sum_l alpha_l * A_l x_l
    where x_h is the global token feature (last in features list).
    Registered as 'multi_layer_aggregator' alignment head.
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
            layer_dims: Feature dims per layer; last is the global layer.
            alphas: Initial/fixed weights for local layers (length n_layers-1).
            align_cfg: Kwargs for CLIPFeatureAlignment (shared across heads).
            learnable_alphas: If True, alpha_l is a learnable parameter; else, buffer.
            init_alpha: Used if `alphas` not provided.
            dtype: Dtype for alpha parameters/buffers initialization.
            assume_inputs_on_device: Skip device moves in forward if True.
        """
        super().__init__()
        layer_dims = list(layer_dims)
        assert len(layer_dims) >= 1, "layer_dims must have at least the global layer"
        self.n_layers = len(layer_dims)
        self.learnable_alphas = learnable_alphas
        self.assume_inputs_on_device = assume_inputs_on_device

        # Init alphas (only for local layers, i.e., everything except the last/global)
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
            # No local layers; keep a tiny buffer for simpler logic (never used)
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
        Aggregate multi-layer features.
        
        Args:
            features: List/tuple of tensors, length == n_layers
        
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
            # Make sure alpha matches compute dtype to avoid upcasts in AMP
            alpha = self.alphas[i].to(local_feat.dtype)
            agg = agg + alpha * local_feat

        return agg