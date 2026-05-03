import os
import logging
import json
from omegaconf import DictConfig, OmegaConf


def log_omegaconf_config(cfg: DictConfig, title: str = "Configuration"):
    """
    Log OmegaConf configuration preserving hierarchy and structure.
    
    Args:
        cfg: OmegaConf DictConfig object
        title: Title for the configuration log
    """
    try:
        logging.info(f"----- {title} -----")
        # Convert to YAML string which preserves hierarchy and is human-readable
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        logging.info(yaml_str)
        logging.info(f"----- End of {title} -----")
    except Exception as e:
        logging.warning(f"Failed to log configuration: {e}")
        # Fallback: try to log as container
        try:
            container = OmegaConf.to_container(cfg, resolve=True)
            logging.info(f"----- {title} (Container Fallback) -----")
            import json
            logging.info(json.dumps(container, indent=2, default=str))
            logging.info(f"----- End of {title} -----")
        except Exception as e2:
            logging.warning(f"Failed to log configuration fallback: {e2}")

def save_omegaconf_config(cfg: DictConfig, save_path: str):
    """
    Save OmegaConf configuration to both YAML and JSON formats.
    
    Args:
        cfg: OmegaConf DictConfig object  
        save_path: Directory path to save configs
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # Save as YAML (preserves structure and comments)
        yaml_path = os.path.join(save_path, "config.yaml")
        with open(yaml_path, "w") as f:
            OmegaConf.save(cfg, f)
        logging.info(f"Configuration saved as YAML: {yaml_path}")
        
        # Save as JSON (for backward compatibility)
        json_path = os.path.join(save_path, "config.json")
        container = OmegaConf.to_container(cfg, resolve=True)
        
        def json_serialize(obj):
            """Custom JSON serializer for non-serializable objects."""
            if hasattr(obj, '__dict__'):
                return str(obj)
            elif isinstance(obj, (set, frozenset)):
                return list(obj)
            else:
                return str(obj)
        
        with open(json_path, "w") as f:
            json.dump(container, f, indent=2, default=json_serialize)
        logging.info(f"Configuration saved as JSON: {json_path}")
        
    except Exception as e:
        logging.warning(f"Failed to save configuration: {e}")


def reconstruct_config_from_args(args) -> DictConfig:
    """
    Reconstruct OmegaConf config from args for saving purposes.
    This is needed because we convert cfg -> args for compatibility.
    """
    try:
        # Create a structured config that matches our schema
        config_dict = {
            "dataset": {
                "name": args.dataset,
                "data_path": args.data_path,
                "subset_name": args.subset_name,
                "split_type": args.split_type,
                "val_ratio": args.val_ratio,
                "split_by_tar": getattr(args, 'split_by_tar', False),
                "use_tar_embeddings": getattr(args, 'use_tar_embeddings', False),
                "use_tar_batching": getattr(args, 'use_tar_batching', False),
            },
            "training": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "eval_n": args.eval_n,
                "force_float32": getattr(args, 'force_float32', False),
                "grad_clip_norm": getattr(args, 'grad_clip_norm', 1.0),
                "save_path": args.save_path,
                "load_checkpoint": getattr(args, 'load_checkpoint', None),
                "use_best_model": getattr(args, 'use_best_model', True),
                "name": args.name,
                "seed": args.seed,
                "test": getattr(args, 'test', False),
            },
            "optimizer": {
                "type": "adam",  # This would need to be inferred or stored
                "learning_rate": args.learning_rate,
                "lr_scaling": getattr(args, 'lr_scaling', "linear"),
                "scheduler": getattr(args, 'scheduler', "cosine"),
                "ft_start_epoch": getattr(args, 'ft_start_epoch', 0),
            },
            "model": {
                "clip_model": args.clip_model,
                "embedding_dim": args.embedding_dim,
                "mlp_layers": getattr(args, 'mlp_layers', 1),
                "alphas": getattr(args, 'alphas', [1.0]),
                "init_alpha": getattr(args, 'init_alpha', 1.0),
                "learnable_alphas": args.learnable_alphas,
                "text_layer_names": args.text_layer_names,
                "image_layer_names": args.image_layer_names,
            },
            "alignment": {
                "alignment_type": args.alignment_type,
                "align_image": args.align_image,
                "align_text": args.align_text,
                "ft_image": getattr(args, 'ft_image', False),
                "ft_text": getattr(args, 'ft_text', False),
            },
            "loss": {
                "w_norm": args.w_norm,
                "w_ortho": args.w_ortho,
                "w_dist": args.w_dist,
                "w_unif": args.w_unif,
                "w_margin": args.w_margin,
                "w_decor": args.w_decor,
                "projection_matrix_path": getattr(args, 'projection_matrix_path', None),
            },
            "dist": {
                "distributed": getattr(args, 'distributed', False),
                "data_parallel": getattr(args, 'data_parallel', False),
                "local_rank": getattr(args, 'local_rank', -1),
                "master_port": getattr(args, 'master_port', None),
                "num_workers": args.num_workers,
                "pin_memory": args.pin_memory,
                "disk_gather": getattr(args, 'disk_gather', True),
            },
            "cache": {
                "cache_folder": args.cache_folder,
                "embedding_path": getattr(args, 'embedding_path', None),
                "embedding_batch_size": getattr(args, 'embedding_batch_size', None),
                "recompute_cache": getattr(args, 'recompute_cache', False),
            },
            "evaluation": {
                "enable_dataset_eval": getattr(args, 'enable_dataset_eval', False),
                "dataset_eval_datasets": getattr(args, 'dataset_eval_datasets', []),
                "dataset_eval_csv_path": getattr(args, 'dataset_eval_csv_path', None),
            }
        }
        
        return OmegaConf.create(config_dict)
        
    except Exception as e:
        logging.warning(f"Failed to reconstruct config from args: {e}")
        # Fallback: create a simple config with just the args
        return OmegaConf.create({"args": vars(args)})
