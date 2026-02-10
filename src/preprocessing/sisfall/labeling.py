from __future__ import annotations

import numpy as np


IMPACT_HALF_WINDOW_SECONDS = 0.5
OVERLAP_THRESHOLD = 0.30


def find_impact_index(signal: np.ndarray) -> int:
    """
    Identify impact point as the max acceleration magnitude sample.
    Expects signal shape (T, 3), in acceleration units (e.g., g).
    """
    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError(f"Expected signal shape (T, 3). Got {signal.shape}")
    
    mag = np.linalg.norm(signal, axis=1)
    return int(np.argmax(mag))


def _overlap_fraction(
    win_start: int, win_end: int,
    reg_start: int, reg_end: int,
) -> float:
    
    overlap = max(0, min(win_end, reg_end) - max(win_start, reg_start))
    win_len = max(1, win_end - win_start)
    
    return overlap / win_len


def impact_window_labels(
    start_indices: np.ndarray,
    window_size: int,
    n_samples: int,
    is_fall: bool,
    impact_index: int | None,
    fs_hz: int,
    impact_half_window_seconds: float = IMPACT_HALF_WINDOW_SECONDS,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> np.ndarray:
    """
    Label windows using overlap with impact-centered region.

    Rules:
    - ADL files: all zeros
    - Fall files: window=1 if overlap(window, impact_region) >= overlap_threshold else 0
    """
    # Checks
    if start_indices.ndim != 1:
        raise ValueError("start_indices must be 1D")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    if not (0.0 <= overlap_threshold <= 1.0):
        raise ValueError("overlap_threshold must be in [0, 1]")
    
    if impact_half_window_seconds < 0:
        raise ValueError("impact_half_window_seconds must be non-negative")

    labels = np.zeros(start_indices.shape[0], dtype=np.int8)
    if not is_fall:
        return labels
    if impact_index is None:
        raise ValueError("impact_index is required for fall files")

    # Calculate params for labelling
    half_w = int(round(impact_half_window_seconds * fs_hz))
    reg_start = max(0, impact_index - half_w)
    reg_end = min(n_samples, impact_index + half_w)
    
    if reg_end <= reg_start:
        reg_end = min(n_samples, reg_start + 1)

    # Main labelling logic
    for i, start in enumerate(start_indices.tolist()):
        end = start + window_size
        frac = _overlap_fraction(start, end, reg_start, reg_end)
        labels[i] = 1 if frac >= overlap_threshold else 0

    return labels
