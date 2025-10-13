import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split

def split_data_custom(metadata_df: pd.DataFrame, val_size: float, seed: int) -> dict:
    """
    Splits the metadata DataFrame into training, validation, and test sets
    based on the specified criteria.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata.
        val_size (float): Proportion of the training data to use for validation.
        seed (int): Seed for random number generation for reproducibility.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test', each containing a DataFrame
              with the corresponding split of the metadata.
    """
    # Training set: All ADL trials from young adults
    train_val_df = metadata_df[(metadata_df["group"] == "Adult") & (metadata_df["is_fall"] == 0)]
    
    # Split training data into training and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=seed)

    # Test set: All fall trials (young + elderly) + ADL data from elderly group
    test_falls = metadata_df[metadata_df["is_fall"] == 1]
    test_elderly_adl = metadata_df[(metadata_df["group"] == "Elderly") & (metadata_df["is_fall"] == 0)]
    test_df = pd.concat([test_falls, test_elderly_adl])

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }
