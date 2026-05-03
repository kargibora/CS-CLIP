import logging
import sys

import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from alignment.ft_experiment import handle_distributed_training, main_with_cfg
from utils.dist import is_main_process
from utils.omega import log_omegaconf_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


@hydra.main(version_base=None, config_path="configs", config_name="coco_ft")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg.epochs = cfg.training.epochs
    OmegaConf.set_struct(cfg, True)

    if is_main_process():
        log_omegaconf_config(cfg)

    spawn_config = handle_distributed_training(cfg)
    if spawn_config:
        mp.spawn(
            spawn_config["spawn_func"],
            args=spawn_config["spawn_args"],
            nprocs=spawn_config["world_size"],
            join=True,
        )
        return

    main_with_cfg(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not any(arg.startswith("--config-path") or arg.startswith("--config-name") for arg in sys.argv):
        print("Warning: Consider using Hydra configs instead of command line arguments")
        print("Example: python align.py --config-name=coco_ft")
        print("Falling back to legacy argument parsing...")
        raise SystemExit(1)

    main()
