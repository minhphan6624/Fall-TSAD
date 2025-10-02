import numpy as np
from typing import Tuple

class StandardScaler:
    """
    A custom StandardScaler to normalize data (mean 0, std dev 1).
    It fits on training data and transforms all data splits using
    the training data's statistics.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """
        Calculates the mean and standard deviation from the provided data.
        Args:
            data (np.ndarray): The training data to fit the scaler on.
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
        self.std[self.std == 0] = 1.0 # Avoid division by zero for constant features, though this is rare.

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the data using the fitted mean and standard deviation.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call .fit() first.")
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fits the scaler to the data and then transforms it.
        """
        self.fit(data)
        return self.transform(data)

def apply_normalization(
    train_data: np.ndarray, 
    val_data: np.ndarray, 
    test_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Applies standardization (Z-score normalization) to the datasets.
    The scaler is fitted only on the training data, and then used to transform
    all three datasets to prevent data leakage.

    Args:
        train_data (np.ndarray): The training data.
        val_data (np.ndarray): The validation data.
        test_data (np.ndarray): The test data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]: 
            A tuple containing the normalized train, validation, test data,
            and the fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    
    # Fit on training data and transform all datasets
    train_normalized = scaler.fit_transform(train_data)
    val_normalized = scaler.transform(val_data)
    test_normalized = scaler.transform(test_data)

    return train_normalized, val_normalized, test_normalized, scaler
