import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import logging

from ..datasets.data_loader import get_dataloaders
from select_threshold import calculate_reconstruction_errors 
from ..models.lstm_ae import LSTM_AE

log = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def evaluate_model(cfg: DictConfig):
    """
    Main function to evaluate the model performance on the test set.
    """
    # Ensure the experiment directory is provided
    if not cfg.experiment_dir:
        raise ValueError("Please provide the experiment directory using experiment_dir=path/to/experiment")

    experiment_dir = Path(cfg.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Load the configuration used for training
    try:
        train_cfg = OmegaConf.load(experiment_dir / ".hydra" / "config.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Training config not found in {experiment_dir / '.hydra' / 'config.yaml'}")
    except Exception as e:
        log.error(f"Error loading training config: {e}")
        raise

    device = torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load data (only test loader is needed for evaluation)
    _, _, test_loader = get_dataloaders(train_cfg)

    # Model initialization
    model = LSTM_AE(**train_cfg.model).to(device)

    # Load trained model weights
    model_path = experiment_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model weights not found in {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    log.info(f"Loaded best model from {model_path}")

    # Load the optimal threshold
    threshold_file = experiment_dir / "optimal_threshold.json"
    if not threshold_file.exists():
        raise FileNotFoundError(f"Optimal threshold file not found in {threshold_file}")
    with open(threshold_file, 'r') as f:
        threshold_data = json.load(f)
    optimal_threshold = threshold_data['optimal_threshold']
    log.info(f"Loaded optimal threshold: {optimal_threshold:.4f}")

    # Calculate reconstruction errors on the test set
    log.info("Calculating reconstruction errors on test set...")
    test_errors, test_labels = calculate_reconstruction_errors(model, test_loader, device)

    # Apply threshold to get predictions
    predictions = (test_errors > optimal_threshold).astype(int)

    # Calculate evaluation metrics
    f1 = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    log.info("\n--- Model Evaluation Results ---")
    log.info(f"F1-Score: {f1:.4f}")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall: {recall:.4f}")
    log.info("\nConfusion Matrix:")
    log.info(f"\n{cm}")
    log.info("------------------------------")

    # Save evaluation results
    results_file = experiment_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': cm.tolist(),
            'optimal_threshold': float(optimal_threshold)
        }, f, indent=4)
    log.info(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    evaluate_model()
