import torch
import os
import time
from pathlib import Path
from torch.utils.data import DataLoader
import logging 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Trainer1:
    def __init__(self, model, cfg, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

        self.criterion = torch.nn.MSELoss()  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.learning_rate) 
        self.model_save_path = run_dir / "model.pth"
        self.best_val_loss = float('inf')
        self.metrics_path = run_dir / "metrics.json"
        self.best_val = float('inf')

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        ''' Full training loop '''
        for epoch in range(self.cfg.trainer.epochs):
            log.info(f"Epoch {epoch+1}/{self.cfg.trainer.epochs}")
            start = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            elapsed = time.time() - start
            print(f"Epoch [{epoch+1}/{self.cfg.trainer.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {elapsed:.2f}s")
        
        # Logging
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": time.time()
        }
        with open(self.metrics_path, 'a') as f:
            f.write(f"{record}\n")

        # Checkpoint
        torch.save(self.model.state_dict(), self.run_dir / "last.pt")
        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(self.model.state_dict(), self.run_dir / "best_model.pt")

    def _train_epoch(self, train_loader: DataLoader):
        ''' Train for one epoch '''
        self.model.train()

        total_loss = 0.0
        for data, _ in train_loader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.criterion(outputs, data)
            loss.backward()
            self.optimizer.step()

            bsz = data.size(0)
            total_loss += loss.item() * bsz       # sum of per-sample losses
            total_samples += bsz

        return total_loss / max(1, total_samples)  # true mean per sample

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader):
        ''' Validate for one epoch '''
        self.model.eval()

        total_loss = 0.0
        for data, _ in val_loader:
            data = data.to(self.device)
            
            outputs = self.model(data)

            loss = self.criterion(outputs, data)
        
            bsz = data.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

        return total_loss / max(1, total_samples)  