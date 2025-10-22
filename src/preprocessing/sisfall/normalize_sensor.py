import numpy as np
from sklearn.preprocessing import RobustScaler

def normalize_sensor(data: np.ndarray, scaler):
    """
    Apply per-axis RobustScaler normalization to 3-axis data.
    Each axis is scaled independently.
    """
    scaled = np.zeros_like(data)

    for i in range(data.shape[1]):
        scaled[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
    return scaled

