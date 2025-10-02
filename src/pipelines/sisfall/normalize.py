import numpy as np
from typing import Tuple
from .load_signal import load_signal

from sklearn.preprocessing import StandardScaler
import joblib

def normalize_splits(splits) -> dict:
    """
    Normalizes each data file in the splits using StandardScaler.
    
    Args:
        splits (dict): A dictionary with keys 'train', 'val', 'test', each containing
                       a DataFrame with 'file' and 'label' columns.
    
    Returns:
        dict: A dictionary with the same keys as input, where each value is a list of tuples
              (normalized_data, label).
    """
    scaler = StandardScaler()
    # Concatenate all training data for fitting the scaler
    X_train = np.vstack([load_signal(f) for f in splits["train"]["file"]])
    scaler.fit(X_train)

    # joblib.dump(scaler, "data/processed/sisfall/scaler.pkl")

    def norm_file(f):
        return scaler.transform(load_signal(f))

    normed = {}
    for split, df in splits.items():
        # More efficient iteration over DataFrame rows
        normed[split] = [
            (norm_file(row['path']), row['is_fall'])
            for _, row in df.iterrows()
        ]
