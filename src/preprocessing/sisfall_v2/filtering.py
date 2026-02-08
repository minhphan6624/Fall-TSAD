from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(
    data: np.ndarray,
    fs_hz: float = 200.0,
    cutoff_hz: float = 5.0,
    order: int = 4,
) -> np.ndarray:
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (T, C). Got shape {data.shape}")
    
    if cutoff_hz <= 0 or cutoff_hz >= fs_hz / 2:
        raise ValueError("cutoff_hz must be between 0 and Nyquist")

    nyq = 0.5 * fs_hz
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)
    
    return filtfilt(b, a, data, axis=0).astype(np.float32)
