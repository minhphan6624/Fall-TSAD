from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_OVERLAP = 0.50


def compute_window_geometry(sampling_rate_hz: float, window_seconds: float, overlap: float) -> tuple[int, int]:
    window_length = int(round(window_seconds * sampling_rate_hz))
    stride = max(1, int(round(window_length * (1.0 - overlap))))
    return window_length, stride


def generate_windows(
    trials_df: pd.DataFrame,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP,
) -> tuple[np.ndarray, pd.DataFrame]:
    
    # Initialization
    window_arrays: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    window_id = 0

    for row in trials_df.itertuples(index=False):
        
        # Extract acc. array 
        acc = np.asarray(row.acc, dtype=np.float32)
        window_length, stride = compute_window_geometry(row.sampling_rate_hz, window_seconds, overlap)
        if acc.shape[0] < window_length:
            continue

        # Main windowing logic
        for start_idx in range(0, acc.shape[0] - window_length + 1, stride):
            end_idx = start_idx + window_length
            window_arrays.append(acc[start_idx:end_idx].astype(np.float32, copy=False))
            
            metadata_rows.append(
                {
                    "window_id": window_id,

                    "subject_id": row.subject_id,
                    "activity_id": row.activity_id,
                    "trial_id": row.trial_id,
                    "is_fall": int(row.is_fall),
                    "split": row.split,
                    
                    "sampling_rate_hz": float(row.sampling_rate_hz),
                    
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                }
            )
            window_id += 1

    metadata_df = pd.DataFrame(metadata_rows)
    
    if window_arrays:
        X = np.stack(window_arrays, axis=0).astype(np.float32, copy=False)
    else:
        X = np.empty((0, 0, 3), dtype=np.float32)
    
    return X, metadata_df
