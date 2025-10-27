from pathlib import Path

import torch
from src.trainers.lstm_ae_trainer import LSTMAETrainer
from src.trainers.data_loader import get_dataloaders
from src.models.lstm_ae import LSTM_AE

# Set basic config values
DATA_DIR = "data/processed/sisfall/tsad/final"
RUN_DIR = Path("runs/lstm_ae")
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

# Model config
INPUT_DIM = 6
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2

def main():
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(Path(DATA_DIR), BATCH_SIZE)

    # Model and trainer
    model = LSTM_AE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss()

    trainer = LSTMAETrainer(model, optimizer, criterion, run_dir=RUN_DIR)

    trainer.fit(train_loader, val_loader, epochs=EPOCHS)


if __name__ == "__main__":
    main()