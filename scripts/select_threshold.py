import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import json
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
import logging

from ..src.modeling.data_loader import get_dataloaders
from ..src.models.lstm_ae import LSTM_AE
from ..src.models.lstm_vae import LSTM_VAE 

log = logging.getLogger(__name__)


def calculate_ae_reconstruction_errors(model, data_loader, device):
    """
    Calculates reconstruction errors for an Autoencoder model.
    """
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for data, batch_labels in data_loader:
            data = data.to(device)
            outputs = model(data)
            loss = torch.mean(torch.pow(data - outputs, 2), dim=[1, 2])

            errors.extend(loss.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return np.array(errors), np.array(labels)


def calculate_vae_reconstruction_errors(model, data_loader, device):
    """
    Calculates reconstruction errors for a Variational Autoencoder model.
    For VAE, the reconstruction error is typically just the MSE of the reconstruction.
    """
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for data, batch_labels in data_loader:
            data = data.to(device)
            reconstruction, _, _ = model(data) # VAE returns reconstruction, mu, log_var
            loss = torch.mean(torch.pow(data - reconstruction, 2), dim=[1, 2])

            errors.extend(loss.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return np.array(errors), np.array(labels)


def find_optimal_threshold(errors, labels):
    """
    Finds the optimal threshold that maximizes the F1-score by iterating through possible thresholds and pick one that gives the best F1-score.
    Can be adjusted to focus on false alarm rate by changing the metric to precision or recall.
    """
    best_f1 = -1
    optimal_threshold = -1

    # Sort errors to efficiently iterate through possible thresholds
    sorted_errors = np.sort(errors)

    for threshold in sorted_errors:
        predictions = (errors > threshold).astype(int)

        # Ensure there are both positive and negative predictions to avoid errors in metrics
        if len(np.unique(predictions)) < 2:
            continue

        f1 = f1_score(labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold

    log.info(f"Optimal Threshold: {optimal_threshold:.4f}, Best F1-Score: {best_f1:.4f}")
    return optimal_threshold, best_f1


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def select_threshold(cfg: DictConfig):
    """
    Main function to select the anomaly threshold using the validation set.
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

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load data, only need validation set for threshold selection
    _, val_loader, _ = get_dataloaders(train_cfg)

    # Model initialization based on model name
    model_name = train_cfg.model.name # Assuming 'name' field in model config
    if model_name == "lstm_ae":
        model = LSTM_AE(**train_cfg.model).to(device)
        calculate_errors_func = calculate_ae_reconstruction_errors
    elif model_name == "lstm_vae":
        model = LSTM_VAE(**train_cfg.model).to(device)
        calculate_errors_func = calculate_vae_reconstruction_errors
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Load the best trained model weights
    model_path = experiment_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model weights not found in {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    val_errors, val_labels = calculate_errors_func(model, val_loader, device)

    optimal_threshold, best_f1 = find_optimal_threshold(val_errors, val_labels)

    # Save the optimal threshold within the experiment directory
    threshold_file = experiment_dir / "optimal_threshold.json"
    with open(threshold_file, 'w') as f:
        json.dump({'optimal_threshold': float(optimal_threshold), 'f1_score_at_threshold': float(best_f1)}, f)
    log.info(f"Optimal threshold saved to {threshold_file}")


if __name__ == "__main__":
    select_threshold()
