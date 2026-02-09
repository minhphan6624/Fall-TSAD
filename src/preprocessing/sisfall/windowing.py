from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FS_HZ = 200
WINDOW_SECONDS = 1.0
OVERLAP = 0.5

def _get_window_indices(num_samples: int, window_size: int, stride: int) -> np.ndarray:
    ''' Get start indexes of windows in the signal'''
    if num_samples < window_size:
        return np.array([], dtype=np.int64)
    
    # Returns an array of evenly spaced values within a specified interval.
    # Start = 0, stop = idx of last window, step = stride
    return np.arange(0, num_samples - window_size + 1, stride, dtype=np.int64)


def segment_signal(
    signal: np.ndarray, fs_hz: int = FS_HZ,
    window_seconds: float = WINDOW_SECONDS, overlap: float = OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one trial signal into fixed-size sliding windows.

    Returns:
      windows: shape (N, window_size, C), dtype float32
      start_indices: shape (N,), start indices of each window
    """
    # Can optionally check for all parameters being positive (>0)
    window_size = int(round(fs_hz * window_seconds))
    stride = max(1, int(round(window_size * (1.0 - overlap))))
    if window_size <= 0:
        raise ValueError("computed window_size must be positive")

    if signal.ndim != 2:
        raise ValueError(f"Expected 2D input (T, C). Got shape {signal.shape}")

    # Get the starting indexes of each individual segment/window
    start_indices = _get_window_indices(signal.shape[0], window_size, stride)

    if start_indices.size == 0:
        empty = np.empty((0, window_size, signal.shape[1]), dtype=np.float32)
        return empty, start_indices

    # Concatenate the windows vertically
    windows = np.stack(
        [signal[s : s + window_size] for s in start_indices],
        axis=0,
    ).astype(np.float32, copy=False)
    
    return windows, start_indices


def build_window_metadata(
    start_indices: np.ndarray,
    file_meta: dict[str, Any],
    fs_hz: int = FS_HZ,
    window_seconds: float = WINDOW_SECONDS,
    split: str | None = None,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build one metadata row per window with a lean schema.
    """

    window_size = int(round(fs_hz * window_seconds))
    if window_size <= 0:
        raise ValueError("computed window_size must be positive")


    records = []
    for local_idx, start_idx in enumerate(start_indices.tolist()):

        records.append(
            {
                "window_local_idx": local_idx,
                
                "file_path": str(file_path) if file_path is not None else str(file_meta.get("path", "")),
                "subject": file_meta.get("subject"),
                "activity": file_meta.get("activity"),
                "trial": int(file_meta.get("trial", -1)),
                "is_fall": int(file_meta.get("is_fall", 0)),
                "split": split if split is not None else file_meta.get("split"),

                "start_idx": int(start_idx),
                "end_idx": int(start_idx + window_size),
            }
        )

    return pd.DataFrame.from_records(records)