"""Loss factory for the fine-tuning pipeline."""

import logging
from typing import Any, Callable

import torch.distributed as dist

from alignment.losses import multi_caption_contrastive_loss


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


LOSS_REGISTRY: dict[str, Callable] = {
    "multi_caption": multi_caption_contrastive_loss,
    "multi_caption_contrastive": multi_caption_contrastive_loss,
}


def get_loss_function(loss_name: str) -> Callable:
    normalized_name = loss_name.lower().replace("-", "_")
    if normalized_name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unknown loss function: '{loss_name}'. Available losses: {available}")
    return LOSS_REGISTRY[normalized_name]


def _cfg_to_dict(cfg_section: Any) -> dict[str, Any]:
    if cfg_section is None:
        return {}
    if hasattr(cfg_section, "items"):
        return dict(cfg_section.items())
    return dict(cfg_section)


def create_loss_from_config(cfg) -> tuple[Callable, dict[str, Any]]:
    loss_cfg = getattr(cfg, "loss", None)
    if loss_cfg is None:
        if is_main_process():
            logging.info("No loss config provided, using multi_caption defaults.")
        return multi_caption_contrastive_loss, {}

    loss_type = getattr(loss_cfg, "loss_type", "multi_caption")
    loss_fn = get_loss_function(loss_type)

    exclude_fields = {"loss_type"}
    loss_kwargs = {
        key: value
        for key, value in _cfg_to_dict(loss_cfg).items()
        if key not in exclude_fields
    }

    if "lambda_entities" in loss_kwargs:
        loss_kwargs["lambda_components"] = loss_kwargs.pop("lambda_entities")
    if "entity_loss_type" in loss_kwargs:
        loss_kwargs["component_loss_type"] = loss_kwargs.pop("entity_loss_type")
    if loss_kwargs.get("contrastive_mode") == "with_entities_negatives":
        loss_kwargs["contrastive_mode"] = "with_components_negatives"
    elif loss_kwargs.get("contrastive_mode") == "with_entities":
        loss_kwargs["contrastive_mode"] = "with_components"

    if "lambda_full" not in loss_kwargs:
        loss_kwargs["lambda_full"] = 1.0
    if "lambda_components" not in loss_kwargs:
        loss_kwargs["lambda_components"] = 1.0
    if "component_loss_type" not in loss_kwargs:
        loss_kwargs["component_loss_type"] = "negclip"

    if is_main_process():
        logging.info("Using loss function: %s", loss_type)
        logging.info("Loss kwargs: %s", loss_kwargs)

    return loss_fn, loss_kwargs
