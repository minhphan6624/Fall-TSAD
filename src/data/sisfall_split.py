import pandas as pd
import numpy as np
from typing import Tuple

def split_data_by_subject(
    metadata_df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the metadata DataFrame into training, validation, and test sets
    based on unique subjects, ensuring no subject's data is in more than one set.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata with a 'subject' column.
        train_ratio (float): Proportion of subjects to allocate to the training set.
        val_ratio (float): Proportion of subjects to allocate to the validation set.
                           The remaining subjects will be allocated to the test set.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
            DataFrames for training, validation, and test sets.
    """
    if not np.isclose(train_ratio + val_ratio + (1 - train_ratio - val_ratio), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    np.random.seed(random_state)
    
    unique_subjects = metadata_df["subject"].unique()
    np.random.shuffle(unique_subjects)

    num_subjects = len(unique_subjects)
    num_train = int(num_subjects * train_ratio)
    num_val = int(num_subjects * val_ratio)

    train_subjects = unique_subjects[:num_train]
    val_subjects = unique_subjects[num_train : num_train + num_val]
    test_subjects = unique_subjects[num_train + num_val :]

    train_df = metadata_df[metadata_df["subject"].isin(train_subjects)].reset_index(drop=True)
    val_df = metadata_df[metadata_df["subject"].isin(val_subjects)].reset_index(drop=True)
    test_df = metadata_df[metadata_df["subject"].isin(test_subjects)].reset_index(drop=True)

    return train_df, val_df, test_df
