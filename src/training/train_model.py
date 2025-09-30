import torch
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

from src.training.data_loader import create_dataloaders
from src.training.trainer import Trainer 

from src.models.lstm_ae import LSTM_AE

def train_model():
    """
    Main function to train the model based on configurations specified in config.yaml.
    """

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    processed_data_path = Path(config['data']['processed_path'])
    batch_size = config['training']['batch_size']
    train_loader, val_loader, _, _ = create_dataloaders(processed_data_path, batch_size)

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

    # Loss and optimizer
    loss_fn_name = config['training']['loss_fn']
    if loss_fn_name == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Loss function {loss_fn_name} not supported.")

    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']

    # Check if the optimizer is supported
    # if optimizer_name not in optim.__all__:
    #     raise ValueError(f"Optimizer {optimizer_name} not supported.")

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    # Training loop
    epochs = config['training']['epochs']
    model_save_path = Path(config['model']['save_path'])
    patience = config['training']['patience']

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        model_save_path=model_save_path,
        model_name=model_name,
        patience=patience
    )
    trainer.fit(epochs)

if __name__ == "__main__":
    train_model()
