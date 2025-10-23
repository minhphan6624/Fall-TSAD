from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SisFallDataset(Dataset):
    def __init__(self, split: str, processed_dir: Path):
        data_file = Path(processed_dir / f"{split}.npz")
        data = np.load(data_file, allow_pickle=True)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = int(self.y[idx])
        return X, y

def get_dataloaders(processed_dir: Path, batch_size=64):
    train = SisFallDataset("train", processed_dir)
    val   = SisFallDataset("val", processed_dir)
    test  = SisFallDataset("test", processed_dir)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
