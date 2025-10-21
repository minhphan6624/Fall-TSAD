import numpy as np
from .load_signal import load_signal

from sklearn.preprocessing import RobustScaler
import joblib

def normalize_splits(splits) -> dict:
    """
    Normalizes each data file in the splits using RobustScaler.
    The scaler is fitted only on the ADL data from the training set.
    
    Args:
        splits (dict): A dictionary with keys 'train', 'val', 'test', each containing
                       a DataFrame with 'path', 'is_fall', and other metadata columns.
    
    Returns:
        dict: A dictionary with the same keys as input, where each value is a list of tuples
              (normalized_data, metadata_row).
    """
    scaler = RobustScaler()
    
    # Concatenate only ADL training data for fitting the scaler
    train_adl_df = splits["train"][splits["train"]["is_fall"] == 0]
    X_train_adl = np.vstack([load_signal(f) for f in train_adl_df["path"]])
    scaler.fit(X_train_adl)

    # joblib.dump(scaler, "data/processed/sisfall/scaler.pkl") # Optional: save scaler

    def norm_file(f):
        return scaler.transform(load_signal(f))

    normed = {}
    for split, df in splits.items():
        # Create a list of tuples (normalized_data, metadata_row) for each file in the split
        normed[split] = [
            (norm_file(row['path']), row) # Pass the entire metadata row
            for _, row in df.iterrows()
        ]
    return normed
