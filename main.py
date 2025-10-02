import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.registry import DATASETS, MODELS
from src.trainers.trainer import Trainer
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Working directory : {cfg.hydra.runtime.cwd}")

    dataset = DATASETS[cfg.data.name](**cfg.data.params)
    model = MODELS[cfg.model.name](**cfg.model.params)
    trainer = Trainer(model, cfg.trainer.params) # Need modification

    trainer.fit(dataset)


if __name__ == "__main__":
    main()
