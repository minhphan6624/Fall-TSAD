import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

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
        y = int(self.y[idx])

        return X, y
