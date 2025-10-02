import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Working directory : {cfg.hydra.runtime.cwd}")

    # Example of accessing config parameters
    # dataset_name = cfg.data.name
    # model_name = cfg.model.name
    # trainer_name = cfg.trainer.name

    # 1. Initialize dataset/dataloader using cfg.data
    # 2. Initialize model using cfg.model
    # 3. Initialize and run trainer using cfg.trainer

    log.info("Refactoring in progress. Main entry point is a placeholder.")
    log.info("Please implement your dataset, model, and trainer initialization here.")

if __name__ == "__main__":
    main()
