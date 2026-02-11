# utils/optim_factory.py
# -*- coding: utf-8 -*-
"""
Optimizer factory for CLIP fine-tuning & linear probing.

Key features
------------
- Param grouping:
  * No weight decay for bias & norm layers (LayerNorm, BatchNorm, GroupNorm, etc.)
  * Optional separate LR / WD for "logit_scale" or temperature "t"
  * Optional head/backbone LR multipliers and freezing for linear probing
- Popular optimizers:
  * AdamW (PyTorch), fused AdamW (if available), AdamW 8-bit (bitsandbytes if installed)
  * Lion (built-in minimal implementation)
  * LAMB / AdamP / Adan via torch-optimizer (optional dep; graceful fallback)
- One-line usage:
    opt = create_optimizer(model, args, mode="ft")  # or mode="linear_probe"

Add CLI args with:
    from utils.optim_factory import add_optimizer_args
    parser = add_optimizer_args(parser)
"""

from __future__ import annotations
import inspect
import logging
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

import torch
import torch.nn as nn


# ------------------------- Optional deps -------------------------

def _try_import_bnb():
    try:
        import bitsandbytes as bnb  # type: ignore
        return bnb
    except Exception:
        return None

def _try_import_torch_optimizer():
    try:
        import torch_optimizer as topt  # type: ignore
        return topt
    except Exception:
        return None


# ------------------------- Minimal Lion optimizer -------------------------

class Lion(torch.optim.Optimizer):
    """
    Lion optimizer (Chen et al. 2023), minimal, torch-friendly.
    Reference update: m = beta1 * m + (1 - beta1) * grad
                      p += -lr * sign(m)
    Here we use the popular variant with two betas and weight decay like AdamW-style.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    p.data.add_(p.data, alpha=-wd * lr)  # decoupled WD

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # momentum update
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                # direction with second beta
                d_p = exp_avg.sign()
                exp_avg.mul_(beta2).add_(g, alpha=1 - beta2)

                p.add_(d_p, alpha=-lr)
        return loss


# ------------------------- Config & CLI -------------------------

@dataclass
class OptimConfig:
    optimizer: str = "adamw"           # {'adamw','adamw8bit','fused_adamw','lion','lamb','adan','adamp'}
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8

    # special params
    logit_scale_lr_mult: float = 0.1   # e.g., smaller LR for logit_scale / temperature
    logit_scale_weight_decay: float = 0.0

    # linear probing / differential LRs
    head_lr_mult: float = 1.0
    backbone_lr_mult: float = 1.0
    freeze_backbone: bool = False
    head_patterns: Tuple[str, ...] = ("align", "projection", "text_projection", "visual.proj")
    backbone_patterns: Tuple[str, ...] = ("visual.", "transformer.")

    # behavior
    use_norm_no_decay: bool = True
    use_bias_no_decay: bool = True

    # implementation toggles
    use_8bit: bool = False  # deprecated by explicit 'adamw8bit', kept for convenience
    use_fused: bool = False


def add_optimizer_args(parser):
    """Augment an argparse.ArgumentParser with common optimizer flags."""
    g = parser.add_argument_group("optimizer")
    g.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "fused_adamw", "adamw8bit", "lion", "lamb", "adan", "adamp"],
                   help="Optimizer to use.")
    g.add_argument("--opt_lr", type=float, default=1e-3, help="Base learning rate.")
    g.add_argument("--opt_wd", type=float, default=0.01, help="Weight decay.")
    g.add_argument("--opt_betas", type=float, nargs=2, default=[0.9, 0.98], help="Betas for Adam-like opts.")
    g.add_argument("--opt_eps", type=float, default=1e-8, help="Epsilon for Adam-like opts.")
    g.add_argument("--opt_use_fused", action="store_true", help="Use fused AdamW if available.")
    g.add_argument("--opt_use_8bit", action="store_true", help="Use bitsandbytes 8-bit AdamW if available.")
    g.add_argument("--opt_logit_lr_mult", type=float, default=0.1, help="LR multiplier for logit scale / temperature.")
    g.add_argument("--opt_logit_wd", type=float, default=0.0, help="WD for logit scale / temperature.")
    g.add_argument("--opt_head_lr_mult", type=float, default=1.0, help="LR multiplier for head params.")
    g.add_argument("--opt_backbone_lr_mult", type=float, default=1.0, help="LR multiplier for backbone params.")
    g.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone (linear probing).")
    return parser


# ------------------------- Param grouping helpers -------------------------

_NORM_TYPES = (
    nn.LayerNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)

def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(p in name for p in patterns)

def _separate_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    cfg: OptimConfig,
    mode: str = "ft",
) -> List[Dict]:
    """
    Split params into groups:
      - no_decay (norms & bias) vs decay
      - optional head vs backbone LR multipliers
      - special group for logit_scale / temperature 't'
    Only includes params with requires_grad=True (so freezing works upstream or via cfg.freeze_backbone).
    """
    decay, no_decay = [], []
    logit_like = []  # logit_scale or temperature t
    head_group, backbone_group = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # identify special scalar(s)
        if name.endswith("logit_scale") or name == "t" or name.endswith(".t"):
            logit_like.append((name, p))
            continue

        # backbone freezing (for linear probe)
        if cfg.freeze_backbone and _match_any(name, cfg.backbone_patterns):
            p.requires_grad_(False)
            continue

        # norm/bias no-decay
        is_no_decay = False
        if cfg.use_bias_no_decay and name.endswith(".bias"):
            is_no_decay = True
        if cfg.use_norm_no_decay:
            # Try to detect by module type
            # Retrieve owning module if possible (optional quality-of-life; safe fallback to name check)
            if any(k in name.lower() for k in ("ln", "norm", "bn", "rmsnorm")):
                is_no_decay = True

        if is_no_decay:
            no_decay.append((name, p))
        else:
            decay.append((name, p))

        # head/backbone buckets (for LR multipliers)
        if _match_any(name, cfg.head_patterns):
            head_group.append((name, p))
        elif _match_any(name, cfg.backbone_patterns):
            backbone_group.append((name, p))

    # Build param groups
    groups: List[Dict] = []

    # Base groups (decay/no_decay) with multipliers
    def _make_group(named_params: List[Tuple[str, torch.nn.Parameter]], lr_mult: float, wd: float):
        if not named_params:
            return None
        return {
            "params": [p for _, p in named_params],
            "lr": base_lr * lr_mult,
            "weight_decay": wd,
        }

    # If user provided head/backbone multipliers, we further split:
    if cfg.head_lr_mult != 1.0 or cfg.backbone_lr_mult != 1.0:
        # Head params intersected with decay/no_decay
        head_decay   = [(n, p) for (n, p) in decay    if (n, p) in set(head_group)]
        head_nodecay = [(n, p) for (n, p) in no_decay if (n, p) in set(head_group)]
        bb_decay     = [(n, p) for (n, p) in decay    if (n, p) in set(backbone_group)]
        bb_nodecay   = [(n, p) for (n, p) in no_decay if (n, p) in set(backbone_group)]
        other_decay   = [(n, p) for (n, p) in decay    if (n, p) not in set(head_group) and (n, p) not in set(backbone_group)]
        other_nodecay = [(n, p) for (n, p) in no_decay if (n, p) not in set(head_group) and (n, p) not in set(backbone_group)]

        for payload, mult, wd in [
            (head_decay,   cfg.head_lr_mult,     weight_decay),
            (head_nodecay, cfg.head_lr_mult,     0.0),
            (bb_decay,     cfg.backbone_lr_mult, weight_decay),
            (bb_nodecay,   cfg.backbone_lr_mult, 0.0),
            (other_decay,  1.0,                  weight_decay),
            (other_nodecay,1.0,                  0.0),
        ]:
            g = _make_group(payload, mult, wd)
            if g: groups.append(g)
    else:
        # Simple two-group split
        g1 = _make_group(decay,    1.0, weight_decay)
        g2 = _make_group(no_decay, 1.0, 0.0)
        if g1: groups.append(g1)
        if g2: groups.append(g2)

    # Special group for logit-like scalars
    if logit_like:
        groups.append({
            "params": [p for _, p in logit_like],
            "lr": base_lr * cfg.logit_scale_lr_mult,
            "weight_decay": cfg.logit_scale_weight_decay,
        })

    # Log groups summary
    if logging.getLogger().isEnabledFor(logging.INFO):
        total = sum(p.numel() for g in groups for p in g["params"])
        logging.info(f"[optim_factory] Built {len(groups)} param groups (trainable params: {total:,}).")
        for i, g in enumerate(groups):
            cnt = sum(p.numel() for p in g["params"])
            logging.info(f"  - group#{i}: lr={g['lr']:.2e}, wd={g['weight_decay']:.2e}, params={cnt:,}")
    return groups


# ------------------------- Optimizer builder -------------------------

def _supports_fused_adamw() -> bool:
    sig = inspect.signature(torch.optim.AdamW.__init__)
    return "fused" in sig.parameters

def _build_optimizer_from_name(name: str, param_groups: List[Dict], cfg: OptimConfig):
    name = name.lower()

    # AdamW family
    if name in ("adamw", "fused_adamw", "adamw8bit"):
        if name == "adamw8bit" or cfg.use_8bit:
            bnb = _try_import_bnb()
            if bnb is None:
                logging.warning("[optim_factory] bitsandbytes not available; falling back to torch AdamW.")
            else:
                return bnb.optim.AdamW8bit(
                    param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0  # wd in param groups
                )

        fused_ok = _supports_fused_adamw()
        use_fused = (name == "fused_adamw" or cfg.use_fused) and fused_ok
        if (name == "fused_adamw" or cfg.use_fused) and not fused_ok:
            logging.warning("[optim_factory] fused AdamW not supported by this torch build; using regular AdamW.")
        return torch.optim.AdamW(
            param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0, fused=use_fused if fused_ok else False
        )

    # Lion (local implementation)
    if name == "lion":
        return Lion(param_groups, lr=cfg.lr, betas=cfg.betas, weight_decay=0.0)

    # torch-optimizer bucket
    topt = _try_import_torch_optimizer()
    if name == "lamb":
        if topt is None:
            logging.warning("[optim_factory] torch-optimizer not found; falling back to AdamW.")
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)
        return topt.Lamb(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)
    if name == "adamp":
        if topt is None:
            logging.warning("[optim_factory] torch-optimizer not found; falling back to AdamW.")
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)
        return topt.AdamP(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)
    if name == "adan":
        if topt is None:
            logging.warning("[optim_factory] torch-optimizer not found; falling back to AdamW.")
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)
        return topt.Adan(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)

    # Fallback
    logging.warning(f"[optim_factory] Unknown optimizer '{name}', using AdamW.")
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=0.0)


def create_optimizer(
    model: nn.Module,
    args_or_cfg,
    mode: str = "ft",
    base_model: Optional[nn.Module] = None,
):
    """
    Build an optimizer for CLIP FT or linear probing.

    Parameters
    ----------
    model : nn.Module
        The (possibly DDP-wrapped) model used for training.
    args_or_cfg : argparse.Namespace | OptimConfig
        Source of optimizer hyperparams.
    mode : {'ft','linear_probe'}
        Controls default behavior for backbone freezing and LR multipliers.
    base_model : nn.Module, optional
        If you pass a wrapper elsewhere (e.g., DDP), you can pass the
        underlying base model here for name patterns; otherwise 'model' is used.

    Returns
    -------
    optimizer : torch.optim.Optimizer
    """
    # Normalize config
    if isinstance(args_or_cfg, OptimConfig):
        cfg = args_or_cfg
    else:
        cfg = OptimConfig(
            optimizer=getattr(args_or_cfg, "optimizer", "adamw"),
            lr=getattr(args_or_cfg, "opt_lr", getattr(args_or_cfg, "learning_rate", 1e-3)),
            weight_decay=getattr(args_or_cfg, "opt_wd", 0.01),
            betas=tuple(getattr(args_or_cfg, "opt_betas", (0.9, 0.98))),
            eps=getattr(args_or_cfg, "opt_eps", 1e-8),
            logit_scale_lr_mult=getattr(args_or_cfg, "opt_logit_lr_mult", 0.1),
            logit_scale_weight_decay=getattr(args_or_cfg, "opt_logit_wd", 0.0),
            head_lr_mult=getattr(args_or_cfg, "opt_head_lr_mult", 1.0),
            backbone_lr_mult=getattr(args_or_cfg, "opt_backbone_lr_mult", 1.0),
            freeze_backbone=bool(getattr(args_or_cfg, "freeze_backbone", False)),
            use_8bit=bool(getattr(args_or_cfg, "opt_use_8bit", False)),
            use_fused=bool(getattr(args_or_cfg, "opt_use_fused", False)),
        )

    # Mode defaults
    if mode == "linear_probe":
        cfg.freeze_backbone = True if getattr(args_or_cfg, "freeze_backbone", None) is None else cfg.freeze_backbone
        cfg.head_lr_mult = max(cfg.head_lr_mult, 1.0)  # typically >= 1
        cfg.backbone_lr_mult = min(cfg.backbone_lr_mult, 1.0)  # typically <= 1

    target_model = base_model if base_model is not None else model

    # Build param groups with decoupled WD in the groups
    param_groups = _separate_param_groups(
        target_model, base_lr=cfg.lr, weight_decay=cfg.weight_decay, cfg=cfg, mode=mode
    )

    # Instantiate optimizer by name
    optimizer = _build_optimizer_from_name(cfg.optimizer, param_groups, cfg)
    return optimizer
