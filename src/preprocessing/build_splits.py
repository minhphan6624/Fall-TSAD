from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 7
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)

def compute_target_sizes(n_subjects: int) -> tuple[int, int]:
    ''' COmpute the number of subjects needed for each split based on a split ratio'''
    n_val = max(1, int(round(n_subjects * DEFAULT_SPLIT_RATIOS[1])))
    n_test = max(1, int(round(n_subjects * DEFAULT_SPLIT_RATIOS[2])))
    
    if n_val + n_test >= n_subjects:
        n_test = max(1, n_subjects - n_val - 1)
    
    return n_val, n_test


def build_subject_summary(trials_df: pd.DataFrame) -> pd.DataFrame:
    ''' Build a summary df for each subject in the dataset'''
    summary = (
        trials_df.groupby("subject_id", sort=True)
        .agg(
            n_trials=("trial_id", "size"), 
            n_fall_trials=("is_fall", "sum")
        ).reset_index()
    )

    summary["n_adl_trials"] = summary["n_trials"] - summary["n_fall_trials"]
    summary["has_fall"] = summary["n_fall_trials"] > 0
    
    return summary.sort_values("subject_id").reset_index(drop=True)

def build_subject_splits(
    summary_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    manual_split_csv: Path | None = None,
) -> pd.DataFrame:
    ''' MAIN METHOD to Build the new subject splits'''

    if manual_split_csv is not None:
        return pd.read_csv(manual_split_csv)

    rng = np.random.default_rng(seed)
    
    # Determine the two types of subjects
    fall_subjects = summary_df.loc[summary_df["has_fall"], "subject_id"].tolist()
    adl_only_subjects = summary_df.loc[~summary_df["has_fall"], "subject_id"].tolist()
    
    fall_subjects = rng.permutation(fall_subjects).tolist()
    adl_only_subjects = rng.permutation(adl_only_subjects).tolist()

    # Assign subjects to splits
    val_subjects = fall_subjects[:1]
    test_subjects = fall_subjects[1:2]
    remaining = fall_subjects[2:] + adl_only_subjects

    n_val, n_test = compute_target_sizes(len(summary_df))

    # Keep adding subjects to val and test split until it reaches the number needed
    while len(val_subjects) < n_val and remaining:
        val_subjects.append(remaining.pop(0))
    while len(test_subjects) < n_test and remaining:
        test_subjects.append(remaining.pop(0))

    train_subjects = remaining # The remaining subjects go to train set


    rows: list[dict[str, object]] = []

    for split_name, subject_ids in (
        ("train", train_subjects),
        ("val", val_subjects),
        ("test", test_subjects),
    ):
        for order_idx, subject_id in enumerate(subject_ids):
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split_name,
                    "split_order": order_idx,
                    "split_seed": seed,
                }
            )

    return pd.DataFrame(rows)

def build_split_artifacts(
    trials_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    manual_split_csv: Path | None = None,
):
    ''' High-level script to perform all 3 processes'''  
    
    # --- Creating summary and split
    subject_summary = build_subject_summary(trials_df)
    
    subject_splits = build_subject_splits(
        summary_df=subject_summary, seed=seed,
        manual_split_csv=manual_split_csv,
    )

    # Merge
    trials_with_split = trials_df.merge(subject_splits[["subject_id", "split"]], on="subject_id", how="left")
    
    return subject_summary, subject_splits, trials_with_split


def save_split_artifacts(
    subject_summary: pd.DataFrame,
    subject_splits: pd.DataFrame,
    trials_with_split: pd.DataFrame,
    out_dir: Path,
):
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subject_summary.to_csv(out_dir / "subject_summary.csv", index=False)
    subject_splits.to_csv(out_dir / "subject_splits.csv", index=False)
    trials_with_split.to_pickle(out_dir / "trials_with_split.pkl")
