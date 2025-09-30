import torch
import yaml
import json
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

from src.training.data_loader import create_dataloaders
from src.models.lstm_ae import LSTM_AE
from src.training.select_threshold import calculate_reconstruction_errors 

def evaluate_model():
    """
    Main function to evaluate the model performance on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data (only test loader is needed for evaluation)
    processed_data_path = Path(config['data']['processed_path'])
    batch_size = config['training']['batch_size']
    _, _, test_loader, _ = create_dataloaders(processed_data_path, batch_size)

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

    # Load trained model weights
    model_save_path = Path(config['model']['save_path'])
    model_path = model_save_path / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded best model from {model_path}")

    # Load the optimal threshold
    threshold_save_path = Path(config['results']['threshold_path'])
    threshold_file = threshold_save_path / "optimal_threshold.json"
    with open(threshold_file, 'r') as f:
        threshold_data = json.load(f)
    optimal_threshold = threshold_data['optimal_threshold']
    print(f"Loaded optimal threshold: {optimal_threshold:.4f}")

    # Calculate reconstruction errors on the test set
    print("Calculating reconstruction errors on test set...")
    test_errors, test_labels = calculate_reconstruction_errors(model, test_loader, device)

    # Apply threshold to get predictions
    predictions = (test_errors > optimal_threshold).astype(int)

    # Calculate evaluation metrics
    f1 = f1_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    print("\n--- Model Evaluation Results ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("------------------------------")

if __name__ == "__main__":
    evaluate_model()
