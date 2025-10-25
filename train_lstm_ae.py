from pathlib import Path

import torch
from src.trainers.lstm_ae_trainer import LSTMAETrainer
from src.trainers.data_loader import get_dataloaders
from src.models.lstm_ae import LSTM_AE

# Set basic config values
DATA_DIR = "data/processed/sisfall/final_tsad"
RUN_DIR = Path("runs/lstm_ae")
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

def main():
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(Path(DATA_DIR), BATCH_SIZE)

    # Model and trainer
    model = LSTM_AE(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss()

    trainer = LSTMAETrainer(model, optimizer, criterion, run_dir=RUN_DIR)

    trainer.fit(train_loader, val_loader, epochs=EPOCHS)


if __name__ == "__main__":
    main()