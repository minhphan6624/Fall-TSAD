import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
import json

from torch.utils.tensorboard import SummaryWriter

class LSTMAETrainer:
    def __init__(self, model, optimizer, criterion, run_dir: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        print(f"Using device: {self.device}")
        self.model = model.to(device)

        self.optimizer = optimizer
        self.criterion = criterion

        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

    def _train_epoch(self, train_loader: DataLoader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for data, _ in train_loader:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.criterion(outputs, data)
            loss.backward()
            self.optimizer.step()

            # Calculate per-sample loss
            total_loss += loss.item() * len(data)
            total_samples += len(data)

        return total_loss / max(1, total_samples)  # true mean per sample

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for data, _ in val_loader:
            data = data.to(self.device)

            outputs = self.model(data)

            loss = self.criterion(outputs, data)

            # Calculate per-sample loss
            total_loss += loss.item() * len(data)
            total_samples += len(data)

        return total_loss / max(1, total_samples)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        writer = SummaryWriter(self.run_dir)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print("Starting epoch: ", epoch+1)
            start = time.time()
            
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            elapsed = time.time() - start

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.2f}s")
            
            # Model Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.run_dir / "best.pt")

            # Save training logs
            record = {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": elapsed
            }
            with open(self.run_dir / "training_logs.jsonl", 'a') as f:
                json.dump(record, f)
                f.write("\n")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
        
        torch.save(self.model.state_dict(), self.run_dir / "last.pt")
        writer.close()
            

    