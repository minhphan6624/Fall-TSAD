from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 7
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_N_FOLDS = 5
DEFAULT_SPLIT_PROTOCOL = "default"
KFOLD_SPLIT_PROTOCOL = "subject_kfold"

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


def validate_subject_splits(summary_df: pd.DataFrame, subject_splits: pd.DataFrame) -> None:
    expected_subjects = set(summary_df["subject_id"])
    split_subjects = subject_splits["subject_id"].tolist()

    duplicated = subject_splits["subject_id"][subject_splits["subject_id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Subjects assigned more than once: {sorted(set(duplicated))}")

    missing = expected_subjects - set(split_subjects)
    extra = set(split_subjects) - expected_subjects
    if missing:
        raise ValueError(f"Subjects missing from split assignment: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown subjects in split assignment: {sorted(extra)}")

    required_splits = {"train", "val", "test"}
    observed_splits = set(subject_splits["split"])
    missing_splits = required_splits - observed_splits
    if missing_splits:
        raise ValueError(f"Missing required splits: {sorted(missing_splits)}")


def build_subject_kfold_splits(
    summary_df: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_index: int = 0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build one subject-wise k-fold train/val/test assignment."""

    rng = np.random.default_rng(seed)
    fall_subjects = summary_df.loc[summary_df["has_fall"], "subject_id"].tolist()
    adl_only_subjects = summary_df.loc[~summary_df["has_fall"], "subject_id"].tolist()

    fall_subjects = rng.permutation(fall_subjects).tolist()
    adl_only_subjects = rng.permutation(adl_only_subjects).tolist()

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for idx, subject_id in enumerate(fall_subjects):
        folds[idx % n_folds].append(subject_id)
    for subject_id in adl_only_subjects:
        target_fold = min(range(n_folds), key=lambda idx: (len(folds[idx]), idx))
        folds[target_fold].append(subject_id)

    val_fold = (fold_index + 1) % n_folds

    rows: list[dict[str, object]] = []
    for fold_id, subject_ids in enumerate(folds):
        split = "test" if fold_id == fold_index else "val" if fold_id == val_fold else "train"
        for order_idx, subject_id in enumerate(subject_ids):
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "split_order": order_idx,
                    "split_seed": seed,
                    "split_protocol": KFOLD_SPLIT_PROTOCOL,
                    "fold_index": fold_index,
                    "fold_id": fold_id,
                    "n_folds": n_folds,
                }
            )

    subject_splits = pd.DataFrame(rows).sort_values(["split", "fold_id", "subject_id"]).reset_index(drop=True)
    validate_subject_splits(summary_df, subject_splits)
    return subject_splits


def build_subject_splits(
    summary_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    manual_split_csv: Path | None = None,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_index: int = 0,
) -> pd.DataFrame:
    ''' MAIN METHOD to Build the new subject splits'''

    if manual_split_csv is not None:
        subject_splits = pd.read_csv(manual_split_csv)
        validate_subject_splits(summary_df, subject_splits)
        return subject_splits

    if split_protocol == KFOLD_SPLIT_PROTOCOL:
        return build_subject_kfold_splits(
            summary_df=summary_df,
            n_folds=n_folds,
            fold_index=fold_index,
            seed=seed,
        )

    if split_protocol != DEFAULT_SPLIT_PROTOCOL:
        raise ValueError(f"Unsupported split_protocol: {split_protocol}")

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
                    "split_protocol": DEFAULT_SPLIT_PROTOCOL,
                    "fold_index": -1,
                    "fold_id": -1,
                    "n_folds": -1,
                }
            )

    subject_splits = pd.DataFrame(rows)
    validate_subject_splits(summary_df, subject_splits)
    return subject_splits

def build_split_artifacts(
    trials_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    manual_split_csv: Path | None = None,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_index: int = 0,
):
    ''' High-level script to perform all 3 processes'''  
    
    # --- Creating summary and split
    subject_summary = build_subject_summary(trials_df)
    
    subject_splits = build_subject_splits(
        summary_df=subject_summary,
        seed=seed,
        manual_split_csv=manual_split_csv,
        split_protocol=split_protocol,
        n_folds=n_folds,
        fold_index=fold_index,
    )

    # Merge
    trials_with_split = trials_df.merge(subject_splits[["subject_id", "split"]], on="subject_id", how="left")
    if trials_with_split["split"].isna().any():
        missing_subjects = sorted(trials_with_split.loc[trials_with_split["split"].isna(), "subject_id"].unique())
        raise ValueError(f"Trials found for subjects without split assignments: {missing_subjects}")
    
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
