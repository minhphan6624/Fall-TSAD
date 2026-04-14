from __future__ import annotations

import numpy as np
import pandas as pd


def training_mask(metadata_df: pd.DataFrame, mode: str) -> np.ndarray:
    ''' Mask data based on training mode (e.g. TSAD only trains on normal)'''
    
    base_mask = metadata_df["split"].to_numpy() == "train"
    
    if mode == "classification":
        return base_mask
    if mode == "tsad":
        return base_mask & (metadata_df["window_label"].to_numpy() == 0)
    
    raise ValueError(f"Unsupported normalization mode: {mode}")


def fit_zscore_stats(X: np.ndarray, metadata_df: pd.DataFrame, mode: str) -> dict[str, np.ndarray]:
    ''' Fit zscore normalization stats on a df'''
    mask = training_mask(metadata_df, mode=mode)
    if not np.any(mask):
        raise ValueError(f"No eligible training windows found to fit {mode} normalization.")

    train_values = X[mask]
    mean = train_values.mean(axis=(0, 1))
    std = train_values.std(axis=(0, 1))
    std = np.where(std == 0.0, 1.0, std)
    
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X: np.ndarray, mean, std) -> np.ndarray:
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)
    normalized = (X - mean) / std

    return normalized.astype(np.float32, copy=False)
