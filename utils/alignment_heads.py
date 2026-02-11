# ============================================
# alignment_heads.py  (new heads + small glue)
# ============================================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
# ---------- utilities ----------
def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# ---------- base interface ----------
class BaseAlignmentHead(nn.Module):
    """Interface so heads are swappable in MultiLayerCLIPAggregator."""
    def __init__(self, in_dim: int, out_dim: Optional[int] = None, normalize: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.normalize = normalize
        self._dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _maybe_norm(self, y: torch.Tensor) -> torch.Tensor:
        return _l2_normalize(y) if self.normalize else y


# ============================================================
# 1) LOW-RANK LINEAR  (A = U V^T)   -- rank r << d
# ============================================================
class LowRankLinearAlignment(BaseAlignmentHead):
    """
    A(x) = U (V^T x),  with U in R^{out_dim x r}, V in R^{in_dim x r}
    If out_dim != in_dim it's still valid (rectangular U, V).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        rank: int = 64,
        normalize: bool = True,
        orthogonal_factors: bool = False,  # if True, init U,V with orthonormal cols
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(in_dim, out_dim, normalize, dtype)
        assert rank > 0 and rank <= min(in_dim, self.out_dim), "rank must be <= min(in_dim, out_dim)"
        self.rank = rank

        self.U = nn.Parameter(torch.empty(self.out_dim, rank, dtype=dtype))
        self.V = nn.Parameter(torch.empty(self.in_dim,  rank, dtype=dtype))

        if orthogonal_factors:
            # Orthogonal columns (QR) init
            with torch.no_grad():
                qU, _ = torch.linalg.qr(torch.randn(self.out_dim, rank, dtype=dtype))
                qV, _ = torch.linalg.qr(torch.randn(self.in_dim,  rank, dtype=dtype))
                self.U.copy_(qU)
                self.V.copy_(qV)
        else:
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> y = x @ (V @ U^T) with associativity:
        # (x @ V) @ U^T  gives [B, r] @ [r, out_dim] = [B, out_dim]
        z = x @ self.V             # [B, r]
        y = z @ self.U.T           # [B, out_dim]
        return self._maybe_norm(y)


# ============================================================
# 2) STRUCTURED BLOCK-DIAGONAL
# ============================================================
class BlockDiagonalLinearAlignment(BaseAlignmentHead):
    """
    A = diag(A_1, ..., A_k) acting on pre-specified index groups.
    Each block can be 'dense', 'diagonal', or 'lowrank' (w/ its own rank).
    Args:
        blocks: List of lists of indices (0-based) specifying each block's coordinates in input space.
                If out_dim != in_dim, you must provide 'out_blocks' (one-to-one) for output indices.
        mode_per_block: list of str in {'dense','diagonal','lowrank'}
        rank_per_block: list of int (only used for 'lowrank' blocks)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        blocks: List[List[int]] = None,
        out_blocks: Optional[List[List[int]]] = None,
        mode_per_block: Optional[List[str]] = None,
        rank_per_block: Optional[List[int]] = None,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(in_dim, out_dim, normalize, dtype)
        assert blocks is not None and len(blocks) > 0, "blocks must be a non-empty list of index lists"
        self.blocks = blocks
        self.k = len(blocks)

        if out_blocks is None:
            # default: same indices for in/out (square A)
            assert self.out_dim == self.in_dim, "If out_blocks is None, out_dim must equal in_dim"
            out_blocks = blocks
        else:
            assert len(out_blocks) == self.k, "out_blocks must match number of blocks"
        self.out_blocks = out_blocks

        mode_per_block = mode_per_block or ["dense"] * self.k
        assert len(mode_per_block) == self.k
        self.mode_per_block = mode_per_block

        rank_per_block = rank_per_block or [None] * self.k
        assert len(rank_per_block) == self.k
        self.rank_per_block = rank_per_block

        # one module per block
        self.blocks_modules = nn.ModuleList()
        for b_idx, (in_idx, out_idx, mode, rnk) in enumerate(zip(self.blocks, self.out_blocks, self.mode_per_block, self.rank_per_block)):
            d_in  = len(in_idx)
            d_out = len(out_idx)
            if mode == "dense":
                lin = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
                # near-identity init if sizes match
                if d_in == d_out:
                    with torch.no_grad():
                        lin.weight.zero_()
                        lin.weight.add_(torch.eye(d_in, dtype=dtype))
                else:
                    nn.init.xavier_uniform_(lin.weight)
                self.blocks_modules.append(lin)

            elif mode == "diagonal":
                assert d_in == d_out, "diagonal block requires d_in == d_out"
                scale = nn.Parameter(torch.ones(d_in, dtype=dtype))
                # wrap in small module to look like Linear
                self.blocks_modules.append(_DiagBlock(scale))

            elif mode == "lowrank":
                assert rnk is not None and rnk > 0 and rnk <= min(d_in, d_out)
                self.blocks_modules.append(_LowRankBlock(d_in, d_out, rnk, dtype=dtype))

            else:
                raise ValueError(f"Unknown mode '{mode}' for block {b_idx}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        device = x.device
        y = x.new_zeros((x.size(0), self.out_dim), dtype=x.dtype, device=device)
        for mod, in_idx, out_idx in zip(self.blocks_modules, self.blocks, self.out_blocks):
            xin = x.index_select(dim=1, index=torch.as_tensor(in_idx, device=device))
            yout = mod(xin)
            y[:, out_idx] = yout
        return self._maybe_norm(y)


class _DiagBlock(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(scale)  # [d]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _LowRankBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, dtype: torch.dtype):
        super().__init__()
        self.U = nn.Parameter(torch.empty(d_out, rank, dtype=dtype))
        self.V = nn.Parameter(torch.empty(d_in,  rank, dtype=dtype))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (x @ V) @ U^T
        return (x @ self.V) @ self.U.T
