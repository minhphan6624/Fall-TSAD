import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class LSTMAETrainer:
    def __init__(self, model, run_dir: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()
        
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

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

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size       # sum of per-sample losses
            total_samples += batch_size

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

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(1, total_samples)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        best_val_loss = float('inf')

        for epoch in range(epochs):
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
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": elapsed
            }
            with open(self.run_dir / "training_logs.json", 'a') as f:
                f.write(f"{record}\n")
        
        torch.save(self.model.state_dict(), self.run_dir / "last.pt")

            

    