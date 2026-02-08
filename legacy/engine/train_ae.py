import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np

from src.models.autoencoder import Autoencoder
from src.trainers.trainer import Trainer
from src.utils.set_seed import set_seed

def main(args):
    set_seed(args.seed)

    # --- 1. Load Data ---
    data_dir = Path(args.data_dir)
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")

    X_train = torch.from_numpy(train_data["X"]).float()
    y_train = torch.from_numpy(train_data["y"]).long() # Labels might be needed for other tasks, keep for consistency
    X_val = torch.from_numpy(val_data["X"]).float()
    y_val = torch.from_numpy(val_data["y"]).long()

    # Determine input_dim for the Autoencoder
    # X_train shape: (num_samples, sequence_length, n_features)
    _, seq_len, n_features = X_train.shape
    input_dim = seq_len * n_features

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 2. Initialize Model, Optimizer, Criterion ---
    model = Autoencoder(input_dim=input_dim, latent_dim=args.latent_dim, hidden_dims=args.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Common for Autoencoders

    # --- 3. Setup Run Directory ---
    run_dir = Path(args.run_dir) / f"ae_latent{args.latent_dim}_lr{args.learning_rate}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- 4. Train Model ---
    trainer = Trainer(model, optimizer, criterion, run_dir)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

    print(f"Training completed. Model and logs saved to: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a standard Autoencoder model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_dir", type=str, default="data/processed/sisfall/tsad/final",
                        help="Directory containing processed data (train.npz, val.npz).")
    parser.add_argument("--run_dir", type=str, default="runs/autoencoder",
                        help="Directory to save model checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--latent_dim", type=int, default=32, help="Dimension of the latent space.")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128, 64],
                        help="Hidden layer dimensions for the encoder/decoder. E.g., --hidden_dims 128 64")

    args = parser.parse_args()
    main(args)
