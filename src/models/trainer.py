import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, model_save_path, model_name, scaler):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.scaler = scaler
        self.best_val_loss = float('inf')

        os.makedirs(self.model_save_path, exist_ok=True)

    def _train_epoch(self):
        self.model.train()
        train_loss = 0
        for data, _ in self.train_loader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, data)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path / f"{self.model_name}_best.pth")
                print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
        
        torch.save(self.scaler, self.model_save_path / "scaler.pth")
        print("Training complete.")
