from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class FallWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_split(split_dir, split):
    """Load one processed train/val/test split."""

    split_dir = Path(split_dir)

    meta = pd.read_csv(split_dir / f"window_meta_{split}.csv")

    with np.load(split_dir / f"windows_{split}.npz") as windows:
        X = windows["X"]
        y = windows["y"]

    if len(X) != len(y) or len(y) != len(meta):
        raise ValueError(f"Mismatched artifact lengths for {split_dir}/{split}.")

    return {
        "X": X, 
        "y": y.astype(int), 
        "meta": meta
    }


def load_window_data(dataset, mode, data_root="data/processed"):
    split_dir = Path(data_root) / dataset / mode

    return {
        "train": load_split(split_dir, "train"),
        "val": load_split(split_dir, "val"),
        "test": load_split(split_dir, "test"),
    }

def make_dataloaders(data, batch_size=64, num_workers=0):
    train_ds = FallWindowDataset(data["train"]["X"], data["train"]["y"])
    val_ds = FallWindowDataset(data["val"]["X"], data["val"]["y"])
    test_ds = FallWindowDataset(data["test"]["X"], data["test"]["y"])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )

def label_counts(y):
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}
