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
    # Separate young and elderly participants
    young_participants = metadata_df[metadata_df["group"] == "young"]
    elderly_participants = metadata_df[metadata_df["group"] == "elderly"]

    # Separate ADL and Fall activities for young participants
    young_adl = young_participants[young_participants["is_fall"] == 0]
    young_fall = young_participants[young_participants["is_fall"] == 1]

    # 1. Train set: 80% of young ADL files
    train_adl, val_test_adl = train_test_split(young_adl, test_size=0.2, random_state=seed, stratify=young_adl['subject'])
    train_df = train_adl

    # 2. Validation set: Remaining 20% of young ADL files + 20% of young FALL files
    val_fall, test_fall_remaining = train_test_split(young_fall, test_size=0.8, random_state=seed, stratify=young_fall['subject'])
    val_df = pd.concat([val_test_adl, val_fall])

    # 3. Test set: Remaining 80% of young FALL files + all elderly ADL/FALL files
    test_df = pd.concat([test_fall_remaining, elderly_participants])

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }
