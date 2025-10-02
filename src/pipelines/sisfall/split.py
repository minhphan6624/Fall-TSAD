import pandas as pd
import numpy as np
from typing import Tuple

def split_data_by_subject(metadata_df: pd.DataFrame, split, seed: int) -> dict:
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
        dict: A dictionary with keys 'train', 'val', 'test', each containing a DataFrame
              with the corresponding split of the metadata.
    """
    train_ratio, val_ratio, test_ratio = split

    np.random.seed(seed)
    subjects = metadata_df["subject"].unique()
    np.random.shuffle(subjects)

    num_subjects = len(subjects)

    num_train = int(num_subjects * train_ratio)
    num_val = int(num_subjects * val_ratio)

    # Create splits based on subjects
    train_ids = subjects[:num_train]
    val_ids = subjects[num_train:num_train + num_val]
    test_ids = subjects[num_train + num_val:]

    train_df = metadata_df[metadata_df["subject"].isin(train_ids)].reset_index(drop=True)
    val_df = metadata_df[metadata_df["subject"].isin(val_ids)].reset_index(drop=True)
    test_df = metadata_df[metadata_df["subject"].isin(test_ids)].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }