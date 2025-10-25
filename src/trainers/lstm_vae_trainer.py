import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
import logging

class VAETrainer:
    def __init__(self, model, cfg, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

        self.criterion = torch.nn.MSELoss() # Reconstruction loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.trainer.learning_rate)
        self.model_save_path = run_dir / "model.pth"
        self.best_val_loss = float('inf')
        self.metrics_path = run_dir / "metrics.json"
        self.best_val = float('inf')
        self.beta = cfg.model.beta # KL divergence weight

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(self.cfg.trainer.epochs):
            
            print(f"Epoch {epoch+1}/{self.cfg.trainer.epochs}")
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
                "time": elapsed
            }
            with open(self.metrics_path, 'a') as f:
                f.write(f"{record}\n")

        # Checkpoint
        torch.save(self.model.state_dict(), self.run_dir / "last.pt")
        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(self.model.state_dict(), self.run_dir / "best_model.pt")

    def _train_epoch(self, train_loader: DataLoader):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for data, _ in train_loader:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, mu, log_var = self.model(data)

            reconstruction_loss = self.criterion(reconstruction, data)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            loss = reconstruction_loss + self.beta * kl_divergence
            loss.backward()
            self.optimizer.step()

            bsz = data.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        
        for data, _ in val_loader:
            data = data.to(self.device)
            
            reconstruction, mu, log_var = self.model(data)

            reconstruction_loss = self.criterion(reconstruction, data)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            loss = reconstruction_loss + self.beta * kl_divergence
        
            bsz = data.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

        return total_loss / max(1, total_samples)
