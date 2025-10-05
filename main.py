import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
import logging

from src.datasets.data_loader import get_dataloaders
from src.trainers.trainer import Trainer
from src.models.lstm_ae import LSTM_AE

from src.utils import set_seed

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed) # Needs to implement in utils.py

    # Run dir
    run_dir = Path(f"experiments/{hydra.core.hydra_config.HydraConfig.get().job.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    model = LSTM_AE(**cfg.model)
    trainer = Trainer(model, cfg, run_dir)
    
    train_loader, val_loader, _ = get_dataloaders(cfg)
    trainer.fit(train_loader, val_loader)

    
if __name__ == "__main__":
    main()