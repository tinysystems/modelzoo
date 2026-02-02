import torch
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate

@dataclass
class Main:
    exp: object
    model: torch.nn.Module
    loader: object
    optim: torch.optim.Optimizer
    epochs: int

@hydra.main(config_path="../conf/", config_name="main", version_base='1.2')
def main(cfg: DictConfig):
    cfg = instantiate(cfg)
    # Run experiment
    cfg.exp.run(cfg)

if __name__ == "__main__":
    main()
