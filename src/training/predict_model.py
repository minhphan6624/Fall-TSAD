import torch
import yaml
from pathlib import Path

import numpy as np

from src.training.data_loader import create_dataloaders
from src.models.lstm_ae import LSTM_AE


def predict_model():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data (only test loader is needed for prediction)
    processed_data_path = Path(config['data']['processed_path'])
    batch_size = config['training']['batch_size']
    _, _, test_loader, scaler = create_dataloaders(processed_data_path, batch_size)

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
    model.eval() # Set model to evaluation mode

    # Perform predictions (generating reconstructions)
    reconstructions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            reconstructions.append(outputs.cpu().numpy())

    reconstructions = np.concatenate(reconstructions, axis=0)
    
    # Inverse transform the data since the model was trained on normalized data.
    # The `test_loader` data is already normalized, so the reconstructions are in normalized scale. # We need to convert them back for good thresholding and visualization.
    
    # Flatten the 3D reconstructions (batch, sequence_length, n_features) to 2D for inverse_transform
    original_shape = reconstructions.shape
    reconstructions_flat = reconstructions.reshape(-1, n_features)
    
    # Inverse transform
    reconstructions_original_scale_flat = scaler.inverse_transform(reconstructions_flat)
    reconstructions_original_scale = reconstructions_original_scale_flat.reshape(original_shape)

    print("Prediction complete. Reconstructions in original scale generated.")
    return reconstructions_original_scale

if __name__ == "__main__":
    reconstructed_data = predict_model()
