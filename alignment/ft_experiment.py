import json
import logging
import os
import types

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from alignment.learning_alignment import train_model_multigpu_merged_batch
from data_loading import build_dataset_from_args
from models import CLIPEndToEndPipeline
from utils.checkpoint import get_base_model, load_checkpoint, save_best_model
from utils.clip_wrapper import load_clip_model
from utils.dist import (
    MultiGPUWrapper,
    distributed_train_wrapper,
    is_main_process,
    monitor_nccl_health,
    safe_barrier,
    set_seed,
)
from utils.ft_helpers import (
    _build_ft_dataloaders,
    _prep_clip_for_alignment,
    _probe_layer_dims_from_batch,
    make_optimizer_and_scheduler,
    sync_ddp_params,
)
from utils.head_init import create_model_from_config


def run_ft_clip(dataset, model_clip, preprocess, split_dict, device, cfg, loss_kwargs=None):
    """Run the FT training loop and return the base model plus best checkpoint payload."""
    train_cfg = cfg.training
    align_cfg = cfg.alignment
    dataset_cfg = cfg.dataset
    is_distributed = bool(getattr(cfg.dist, "distributed", False))

    args = types.SimpleNamespace()
    args.dataset = dataset_cfg.name
    args.batch_size = train_cfg.batch_size
    args.learning_rate = cfg.optimizer.learning_rate
    args.distributed = is_distributed
    args.data_parallel = getattr(cfg.dist, "data_parallel", False)

    if loss_kwargs is None:
        loss_kwargs = {}

    train_indices = split_dict["train"]["indices"]
    val_indices = split_dict["val"]["indices"]
    train_dl, val_dl, train_sampler, _ = _build_ft_dataloaders(
        args, dataset, train_indices, val_indices, is_distributed
    )

    image_layer_names = ["final"] + list(getattr(cfg.model, "image_layer_names", []))
    text_layer_names = ["final"] + list(getattr(cfg.model, "text_layer_names", []))
    model_clip, align_dtype = _prep_clip_for_alignment(model_clip, getattr(train_cfg, "force_float32", False))

    image_layer_dims, text_layer_dims = _probe_layer_dims_from_batch(
        train_dl, model_clip, device, image_layer_names, text_layer_names, align_dtype
    )

    head, batch_unpack_fn, loss_fn, evaluate_fn, _ = create_model_from_config(
        cfg=cfg,
        image_layer_dims=image_layer_dims,
        text_layer_dims=text_layer_dims,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        dtype=align_dtype,
        is_ft=True,
    )

    model = CLIPEndToEndPipeline(
        model=model_clip,
        head=head,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        ft_image_encoder=align_cfg.ft_image,
        ft_text_encoder=align_cfg.ft_text,
        assume_inputs_on_device=True,
    ).to(device)

    gpu_wrapper = MultiGPUWrapper(model, args)
    model = gpu_wrapper.get_model()

    if is_distributed and dist.is_available() and dist.is_initialized():
        try:
            sync_ddp_params(gpu_wrapper)
        except NameError:
            pass

    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg, train_dl)
    loss_cfg = cfg.loss

    best_model_dict = train_model_multigpu_merged_batch(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        batch_unpack_fn=batch_unpack_fn,
        evaluate_fn=evaluate_fn,
        cfg=cfg,
        gpu_wrapper=gpu_wrapper,
        train_sampler=train_sampler,
        scheduler=scheduler,
        loss_kwargs=loss_kwargs,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
    )
    return gpu_wrapper.get_base_model(), best_model_dict


def handle_distributed_training(cfg):
    """Support both torchrun and spawn-based multi-GPU entrypoints."""
    from utils.dist import is_distributed_launch_mode

    if is_distributed_launch_mode():
        print("Detected torch.distributed.launch mode")
        return None

    if not cfg.dist.distributed:
        return None

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Distributed requested but only 1 GPU available. Proceeding with single GPU.")
        cfg.dist.distributed = False
        return None

    print("Using multiprocessing.spawn mode")
    for env_var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if env_var in os.environ:
            del os.environ[env_var]
            print(f"Cleared existing {env_var} environment variable")

    if cfg.dist.master_port is not None:
        os.environ["MASTER_PORT"] = str(cfg.dist.master_port)
        print(f"Using specified master port: {cfg.dist.master_port}")
    elif "MASTER_PORT" not in os.environ:
        import random

        master_port = random.randint(12000, 65000)
        os.environ["MASTER_PORT"] = str(master_port)
        print(f"Using random master port: {master_port}")
    else:
        print(f"Using master port from environment: {os.environ['MASTER_PORT']}")

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    print(f"Final MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT_SET')}")
    print(f"Final MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT_SET')}")
    return {
        "world_size": world_size,
        "spawn_func": distributed_train_wrapper,
        "spawn_args": (world_size, cfg, main_with_cfg),
    }


def build_loss_kwargs_from_cfg(cfg, device):
    """Backward-compatible loss kwargs builder for the FT training loop."""
    from alignment.loss_factory import create_loss_from_config

    logging.warning(
        "build_loss_kwargs_from_cfg() is deprecated. Use alignment.loss_factory.create_loss_from_config() instead."
    )
    _, loss_kwargs = create_loss_from_config(cfg)
    return loss_kwargs


def _build_runtime_args(cfg):
    train_cfg = cfg.training
    align_cfg = cfg.alignment
    dataset_cfg = cfg.dataset
    dist_cfg = cfg.dist

    args = types.SimpleNamespace()
    args.dataset = dataset_cfg.name
    args.data_path = dataset_cfg.data_path
    args.subset_name = dataset_cfg.subset_name
    args.val_ratio = dataset_cfg.val_ratio
    args.seed = train_cfg.seed
    args.split_type = dataset_cfg.split_type
    args.num_workers = getattr(dist_cfg, "num_workers", 2)
    args.data_parallel = getattr(dist_cfg, "data_parallel", False)
    args.load_checkpoint = train_cfg.load_checkpoint
    args.distributed = dist_cfg.distributed
    args.batch_size = train_cfg.batch_size
    args.use_best_model = train_cfg.use_best_model
    args.ft_image = align_cfg.ft_image
    args.ft_text = align_cfg.ft_text
    args.save_path = train_cfg.save_path
    args.exp_name = getattr(train_cfg, "name", "experiment")
    args.dataset_kwargs = (
        OmegaConf.to_container(dataset_cfg.dataset_kwargs, resolve=True)
        if hasattr(dataset_cfg, "dataset_kwargs")
        else {}
    )
    return args


def _resolve_device(cfg):
    dist_cfg = cfg.dist
    if hasattr(cfg, "device_str") and cfg.device_str is not None:
        device = torch.device(cfg.device_str)
    elif hasattr(dist_cfg, "local_rank") and dist_cfg.local_rank is not None and dist_cfg.local_rank >= 0:
        device = torch.device(f"cuda:{dist_cfg.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = getattr(dist_cfg, "rank", 0)
    logging.info(f"Rank {rank}: Using device: {device}")
    if device.type == "cuda":
        current_device = torch.cuda.current_device()
        if current_device != device.index:
            logging.warning(
                f"Rank {rank}: Current CUDA device ({current_device}) != expected device ({getattr(device, 'index', None)})"
            )
    return device


def _configure_distributed_launch(cfg):
    from utils.dist import get_distributed_info, is_distributed_launch_mode, setup_distributed

    if not is_distributed_launch_mode():
        return

    rank, local_rank, world_size = get_distributed_info()
    setup_distributed(rank, world_size)
    OmegaConf.set_struct(cfg, False)
    cfg.dist.rank = rank
    cfg.dist.local_rank = local_rank
    cfg.device_str = f"cuda:{local_rank}"
    cfg.dist.distributed = True
    OmegaConf.set_struct(cfg, True)
    torch.cuda.set_device(local_rank)
    logging.info(
        "torch.distributed.launch mode: rank=%s, local_rank=%s, world_size=%s",
        rank,
        local_rank,
        world_size,
    )


def _validate_requested_layers(cfg, clip_model):
    if cfg.model.image_layer_names:
        logging.info(f"Requested image layers: {cfg.model.image_layer_names}")
        available_image_modules = dict(clip_model.visual.named_modules())
        missing_image_layers = [name for name in cfg.model.image_layer_names if name not in available_image_modules]
        if missing_image_layers:
            logging.error(f"Missing image layer names: {missing_image_layers}")
            logging.error("Available image layers (showing some examples):")
            for i, (name, _) in enumerate(clip_model.visual.named_modules()):
                if i < 20:
                    logging.error(f"  - {name}")
                elif i == 20:
                    logging.error("  ... (and more)")
                    break
            raise ValueError(f"Invalid image layer names: {missing_image_layers}")

    if not cfg.model.text_layer_names:
        return

    logging.info(f"Requested text layers: {cfg.model.text_layer_names}")
    available_text_modules = dict(clip_model.transformer.named_modules())
    missing_text_layers = [name for name in cfg.model.text_layer_names if name not in available_text_modules]
    if missing_text_layers:
        logging.error(f"Missing text layer names: {missing_text_layers}")
        logging.error("Available text layers (showing some examples):")
        for i, (name, _) in enumerate(clip_model.transformer.named_modules()):
            if i < 20:
                logging.error(f"  - {name}")
            elif i == 20:
                logging.error("  ... (and more)")
                break
        raise ValueError(f"Invalid text layer names: {missing_text_layers}")


def _create_split_dict(dataset, args):
    logging.info(f"Using standard split for dataset: {args.dataset}")
    return dataset.split_dataset(
        val_ratio=args.val_ratio,
        test_ratio=0.0,
        split_type=args.split_type,
        seed=args.seed,
    )

def _save_run_artifacts(args, cfg, model, best_model_dict):
    if not args.save_path or not is_main_process():
        return

    model_checkpoint_folder = os.path.join(args.save_path, f"{args.exp_name}")
    os.makedirs(model_checkpoint_folder, exist_ok=True)

    save_best_model(model, os.path.join(model_checkpoint_folder, "last_checkpoint.pt"))
    if best_model_dict is not None and best_model_dict.get("best_model_state_dict") is not None:
        torch.save(best_model_dict["best_model_state_dict"], os.path.join(model_checkpoint_folder, "best_checkpoint.pt"))
        logging.info(f"Saved best checkpoint to {model_checkpoint_folder}/best_checkpoint.pt")
    else:
        logging.warning("No best_model_state_dict found, only saving last_checkpoint.pt.")

    with open(os.path.join(model_checkpoint_folder, "config.json"), "w") as f:
        config_dict = {}
        for k, v in vars(args).items():
            try:
                json.dumps(v)
                config_dict[k] = v
            except (TypeError, ValueError):
                config_dict[k] = str(v)
        json.dump(config_dict, f, indent=4)


def main_with_cfg(cfg):
    """Run the FT experiment for either single-GPU or distributed launch mode."""
    train_cfg = cfg.training
    clip_cache_dir = os.environ.get(
        "CLIP_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "clip"),
    )

    _configure_distributed_launch(cfg)
    args = _build_runtime_args(cfg)
    device = _resolve_device(cfg)

    set_seed(train_cfg.seed)

    logging.info(f"Using CLIP model: {cfg.model.clip_model}")
    clip_model, preprocess = load_clip_model(
        model_name=cfg.model.clip_model,
        device=device,
        download_root=clip_cache_dir,
        force_openclip=True,
    )
    logging.info(f"Successfully loaded CLIP model: {cfg.model.clip_model}")
    logging.info(f"Model type: {clip_model.model_type}")
    if hasattr(clip_model, "model_name"):
        logging.info(f"Full model name: {clip_model.model_name}")

    exp_name = f"{train_cfg.name}"
    OmegaConf.set_struct(cfg, False)
    cfg.exp_name = exp_name
    OmegaConf.set_struct(cfg, True)
    logging.info(f"Experiment name: {exp_name}")

    if is_main_process():
        model_checkpoint_folder = os.path.join(train_cfg.save_path, f"{exp_name}")
        os.makedirs(model_checkpoint_folder, exist_ok=True)
        with open(os.path.join(model_checkpoint_folder, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    _validate_requested_layers(cfg, clip_model)
    logging.info(f"Using device: {device}")
    dataset = build_dataset_from_args(args, preprocess)
    logging.info(f"Using dataset: {args.dataset}")

    split_dict = _create_split_dict(dataset, args)
    if args.distributed and dist.is_initialized():
        if is_main_process():
            logging.info(
                f"Split created: train={len(split_dict['train']['indices'])}, val={len(split_dict['val']['indices'])}"
            )
        if not safe_barrier():
            logging.warning("Failed to sync after split creation, continuing anyway")
        else:
            logging.info("All processes synchronized after split creation")

    loss_kwargs = build_loss_kwargs_from_cfg(cfg, device)
    if args.ft_image is False and args.ft_text is False:
        raise ValueError("At least one of ft_image or ft_text must be True for FT alignment.")

    model, best_model_dict = run_ft_clip(dataset, clip_model, preprocess, split_dict, device, cfg, loss_kwargs)

    if args.distributed and dist.is_initialized() and not safe_barrier():
        logging.warning("Failed to sync after FT training, continuing anyway")

    try:
        if args.use_best_model and is_main_process():
            if best_model_dict is not None:
                logging.info("Loading best model from validation loss.")
                base_model = get_base_model(model)
                base_model.load_state_dict(best_model_dict["best_model_state_dict"])
                model.to(device)
            elif args.load_checkpoint:
                logging.info("No best model found, using the last checkpoint.")
                load_checkpoint(model, args.load_checkpoint, device=device)
                model.to(device)
            else:
                logging.warning("No checkpoint provided, using the model as is.")
    except Exception as e:
        logging.error(f"Failed to load best model: {e}")
        logging.warning("Proceeding with current model weights.")

    if args.distributed and dist.is_initialized():
        if not safe_barrier():
            logging.warning("Failed to sync after FT model loading, continuing anyway")
        if not monitor_nccl_health():
            logging.error("NCCL health check failed before FT evaluation")
            logging.warning("Proceeding with FT evaluation despite NCCL issues")

    try:
        _save_run_artifacts(args, cfg, model, best_model_dict)
    except Exception as e:
        if is_main_process():
            logging.error(f"Failed to save checkpoints: {e}")
