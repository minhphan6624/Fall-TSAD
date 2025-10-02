import os
import torch
import yaml
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from src.datasets.data_loader import create_dataloaders
from src.models.lstm_ae import LSTM_AE

def calculate_reconstruction_errors(model, data_loader, device):
    """
    Calculates reconstruction errors for a given model and data loader.
    """
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for data, batch_labels in data_loader:
            data = data.to(device)

            # Calculate Mean Squared Error (MSE) for each sequence in the batch
            # dim=[1, 2] means we average over the sequence length and feature dimensions
            outputs = model(data)
            loss = torch.mean(torch.pow(data - outputs, 2), dim=[1, 2]) 

            # Store errors and corresponding labels
            errors.extend(loss.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return np.array(errors), np.array(labels)

def find_optimal_threshold(errors, labels):
    """
    Finds the optimal threshold that maximizes the F1-score.
    Assumes labels are 0 for normal, 1 for anomaly.
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
    
    print(f"Optimal Threshold: {optimal_threshold:.4f}, Best F1-Score: {best_f1:.4f}")
    return optimal_threshold, best_f1

def select_threshold():
    """
    Main function to select the anomaly threshold using the validation set.
    """

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data, only need validation set for threshold selection
    processed_data_path = Path(config['data']['processed_path'])
    batch_size = config['training']['batch_size']
    _, val_loader, _ = create_dataloaders(processed_data_path, batch_size)

    # Model initialization
    model_name = config['model']['name']
    n_features = config['model']['n_features']
    hidden_dim = config['model']['hidden_dim']
    n_layers = config['model']['n_layers']
    dropout = config['model']['dropout']

    if model_name == 'LSTM_AE':
        model = LSTM_AE(n_features, hidden_dim, n_layers, dropout).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Load the best trained model weights
    model_save_path = Path(config['model']['save_path'])
    model_path = model_save_path / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded best model from {model_path}")

    print("Calculating reconstruction errors on validation set...")
    val_errors, val_labels = calculate_reconstruction_errors(model, val_loader, device)

    print("Finding optimal threshold...")
    optimal_threshold, best_f1 = find_optimal_threshold(val_errors, val_labels)

    # Save the optimal threshold
    threshold_save_path = Path(config['results']['threshold_path'])
    os.makedirs(threshold_save_path, exist_ok=True)
    
    threshold_file = threshold_save_path / "optimal_threshold.json"
    with open(threshold_file, 'w') as f:
        json.dump({'optimal_threshold': float(optimal_threshold), 'f1_score_at_threshold': float(best_f1)}, f)
    print(f"Optimal threshold saved to {threshold_file}")

if __name__ == "__main__":
    select_threshold()
