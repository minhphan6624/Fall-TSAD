import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os

class Trainer:
    """
    A generic trainer class for PyTorch models.
    Handles training loop, validation, early stopping.
    """
    def __init__(self, model, model_name, device, train_loader, val_loader, criterion, optimizer, model_save_path, patience=3):
        
        # Model and device
        self.model = model
        self.device = device
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss fn. and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.patience = patience
        self.early_stopping_counter = 0

    # Run a single epoch of training
    def _train_epoch(self):
        self.model.train() # Set model to training mode
        train_loss = 0

        for data, _ in self.train_loader:
            data = data.to(self.device) # Move data to the computing device
            
            self.optimizer.zero_grad() # Clear gradients
            outputs = self.model(data)

            loss = self.criterion(outputs, data)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_loader)

        return avg_train_loss

    # Run a single epoch of validation
    def _validate_epoch(self):
        self.model.eval() # Set model to evaluation mode
        val_loss = 0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss

    def fit(self, epochs):
        for epoch in range(epochs):
            # Train and validate for one epoch
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path / f"{self.model_name}_best.pth")
                print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
                self.early_stopping_counter = 0 # Reset counter if validation loss improves
            else:
                self.early_stopping_counter += 1 # Increment counter if no improvement
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")

            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {self.patience} consecutive epochs.")
                break
        
        print("Training complete.")
