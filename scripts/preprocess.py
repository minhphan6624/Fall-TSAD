import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.pipelines.sisfall.aa_run_pipeline import run_pipeline
from src.pipelines import DATASETS

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """
    Generic preprocessing entrypoint for different datasets.
    Select the dataset pipeline based on cfg.dataset.name and run it.
    """
    dataset_name = cfg.dataset.name

    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in registry. Available datasets: {list(DATASETS.keys())}")
    
    print(f"Running preprocessing for dataset: {dataset_name}")
    
    dataset_pipeline = DATASETS[dataset_name]   
    run_pipeline(dataset_pipeline, cfg)

if __name__ == "__main__":
    main()


