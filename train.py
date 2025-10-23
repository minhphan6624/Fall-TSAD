from pathlib import Path
from src.trainers.lstm_ae_trainer import LSTMAETrainer
from src.trainers.data_loader import get_dataloaders
from src.models.lstm_ae import LSTM_AE

# Set basic config values
DATA_DIR = "data/processed/sisfall/ready"
BATCH_SIZE = 128
EPOCHS = 50
RUN_DIR = Path("runs/lstm_ae")

def main():
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(Path(DATA_DIR), BATCH_SIZE)

    # Model and trainer
    model = LSTM_AE(n_features=6, hidden_dim=64, num_layers=2, dropout=0.2)

    trainer = LSTMAETrainer(model, run_dir=RUN_DIR)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

if __name__ == "__main__":
    main()