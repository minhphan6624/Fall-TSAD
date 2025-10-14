import hydra
from omegaconf import DictConfig

from pathlib import Path
import logging

from src.datasets.data_loader import get_dataloaders
from src.modeling.trainer import Trainer1, VAETrainer # Import VAETrainer
from src.models.lstm_ae import LSTM_AE
from src.models.lstm_vae import LSTM_VAE # Import LSTM_VAE

from src.utils.set_seed import set_seed

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Hydra's output directory is the run directory
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # Model initialization based on model name
    model_name = cfg.model.name
    if model_name == "lstm_ae":
        model = LSTM_AE(**cfg.model)
        trainer_class = Trainer1
    elif model_name == "lstm_vae":
        model = LSTM_VAE(**cfg.model)
        trainer_class = VAETrainer
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    trainer = trainer_class(model, cfg, run_dir)
    
    train_loader, val_loader, _ = get_dataloaders(cfg)
    log.info(f"Training started with {model_name}...")
    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    main()
