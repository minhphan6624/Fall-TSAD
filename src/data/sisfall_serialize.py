import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from .sisfall_normalize import StandardScaler

def save_processed_data(
    output_dir: Path,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    scaler: StandardScaler,
    metadata_df: pd.DataFrame
):
    """
    Saves the processed (segmented and normalized) data, labels,
    the fitted scaler, and the full metadata DataFrame to disk.

    Args:
        output_dir (Path): The directory where the processed data will be saved.
        train_data (np.ndarray): Normalized and segmented training data.
        train_labels (np.ndarray): Labels for the training data.
        val_data (np.ndarray): Normalized and segmented validation data.
        val_labels (np.ndarray): Labels for the validation data.
        test_data (np.ndarray): Normalized and segmented test data.
        test_labels (np.ndarray): Labels for the test data.
        scaler (StandardScaler): The fitted StandardScaler instance.
        metadata_df (pd.DataFrame): The complete metadata DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data and labels using numpy's savez_compressed
    np.savez_compressed(
        output_dir / "processed_data.npz",
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
    )

    # Save the scaler object
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save the metadata DataFrame
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)

    print(f"Processed data, scaler, and metadata saved to {output_dir}")

def load_processed_data(output_dir: Path) -> dict:
    """
    Loads the processed data, labels, fitted scaler, and metadata DataFrame from disk.

    Args:
        output_dir (Path): The directory from which to load the processed data.

    Returns:
        dict: A dictionary containing the loaded data, labels, scaler, and metadata.
    """
    data = np.load(output_dir / "processed_data.npz")
    
    with open(output_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    metadata_df = pd.read_csv(output_dir / "metadata.csv")

    return {
        "train_data": data["train_data"],
        "train_labels": data["train_labels"],
        "val_data": data["val_data"],
        "val_labels": data["val_labels"],
        "test_data": data["test_data"],
        "test_labels": data["test_labels"],
        "scaler": scaler,
        "metadata_df": metadata_df,
    }
