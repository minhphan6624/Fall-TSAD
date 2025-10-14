from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SisFallDataset(Dataset):
    """
    Custom Dataset for loading time series data.
    """

    def __init__(self, split: str, processed_dir: Path):
        data = np.load(processed_dir / f"{split}_data.npy")
        labels = np.load(processed_dir / f"{split}_labels.npy")

        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        # X = self.X[idx].detach().clone()
        y = int(self.y[idx])

        return X, y


def get_dataloaders(cfg):
    train = SisFallDataset("train", Path(cfg.data.processed_dir))
    val   = SisFallDataset("val", Path(cfg.data.processed_dir))
    test  = SisFallDataset("test", Path(cfg.data.processed_dir))

    train_loader = DataLoader(train, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader   = DataLoader(val, batch_size=cfg.data.batch_size)
    test_loader  = DataLoader(test, batch_size=cfg.data.batch_size)
    return train_loader, val_loader, test_loader
