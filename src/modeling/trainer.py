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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        ''' Full training loop '''
        for epoch in range(self.cfg.trainer.epochs):
            log.info(f"Epoch {epoch+1}/{self.cfg.trainer.epochs}")
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            print(f"Epoch [{epoch+1}/{self.cfg.trainer.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time():.2f}s")
        
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

            total_loss += loss.item() * data.size(0)

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader):
        ''' Validate for one epoch '''
        self.model.eval()

        total_loss = 0.0
        for data, _ in val_loader:
            data = data.to(self.device)
            
            outputs = self.model(data)

            loss = self.criterion(outputs, data)
            
            total_loss += loss.item() * data.size(0)

        return total_loss / len(val_loader)

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

        for data, idx  in self.train_loader:
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
    @torch.no_grad()
    def _validate_epoch(self):
        self.model.eval() # Set model to evaluation mode
        val_loss = 0

        for data, _ in self.val_loader:
            data = data.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, data)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss

    def fit(self, epochs):
        """
        Trains the model for a specified number of epochs,
        """
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train and validate for one epoch
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f}s")

            # If validation loss improves
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss # Update best validation loss

                # Save the best model
                torch.save(self.model.state_dict(), self.model_save_path / f"{self.model_name}_best.pth")
                print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
                
                # Reset early stopping counter
                self.early_stopping_counter = 0 
            else:
                # Increment counter if no improvement
                self.early_stopping_counter += 1 
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")

            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {self.patience} consecutive epochs.")
                break
        
        print("Training complete.")
