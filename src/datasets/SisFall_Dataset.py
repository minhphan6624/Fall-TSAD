import numpy as np
import torch
from torch.utils.data import Dataset

class SisFallDataset(Dataset):
    """
    Custom Dataset for loading time series data.
    """

    def __init__(self, split, processed_dir = "data/processed/sisfall"):
        self.data = np.load(f"{processed_dir}/{split}_data.npz")
        self.X = torch.tensor(self.data["X"], dtype=torch.float32)
        self.y = torch.tensor(self.data["Y"], dtype=torch.long)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return torch.tensor(self.X[idx], dtype=torch.float32), int(self.y[idx])
        return self.X[idx], self.y[idx]