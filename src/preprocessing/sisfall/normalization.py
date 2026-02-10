from __future__ import annotations

import numpy as np


EPS = 1e-8


def fit_zscore_stats(x: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-channel z-score stats from window tensor X with shape (N, T, C).
    Returns:
      mean: shape (C,)
      std: shape (C,)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, C). Got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Cannot fit z-score stats on empty tensor")

    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return mean, std


def apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply per-channel z-score to X shape (N, T, C).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, C). Got {x.shape}")
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("mean/std must be 1D per-channel vectors")
    if x.shape[2] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            f"Shape mismatch. X channels={x.shape[2]}, mean={mean.shape}, std={std.shape}"
        )
    
    return ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32, copy=False)
