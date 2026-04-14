from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

INTERIM_PICKLE_NAMES = {
    "sisfall": "SisFall.pkl",
    "fallalld": "FallAllD.pkl",
    "umafall": "UMAFall.pkl",
    "upfall": "UP-FALL.pkl",
}

TRIAL_COLUMNS = [
    "dataset",
    "subject_id",
    "trial_id",
    "activity_id",
    "is_fall",
    "sampling_rate_hz",
    "n_samples",
    "acc",
    "raw_file",
]

def validate_acc_array(acc: np.ndarray) -> None:
    if not isinstance(acc, np.ndarray):
        raise TypeError("acc must be a numpy array")
    if acc.ndim != 2:
        raise ValueError(f"acc must be 2D, got shape {acc.shape}")
    if acc.shape[1] != 3:
        raise ValueError(f"acc must have shape (T, 3), got {acc.shape}")
    if acc.shape[0] == 0:
        raise ValueError("acc must contain at least one sample")


def validate_trial_row(row: dict) -> None:
    required = [
        "dataset",
        "subject_id",
        "trial_id",
        "activity_id",
        "is_fall",
        "sampling_rate_hz",
        "n_samples",
        "acc",
        "raw_file",
    ]
    for key in required:
        if key not in row:
            raise KeyError(f"Missing required field: {key}")

    validate_acc_array(row["acc"])

# Create a row for a trial
def make_trial_row(
    dataset: str, subject_id: str,
    trial_id: str, activity_id: str,
    is_fall: int | bool, acc: np.ndarray,
    raw_file: str | Path, sampling_rate_hz: int | float,
) -> dict:
    return {
        "dataset": dataset,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "activity_id": activity_id,
        "is_fall": int(is_fall),
        "sampling_rate_hz": float(sampling_rate_hz),
        "n_samples": int(acc.shape[0]),
        "acc": acc,
        "raw_file": str(raw_file),
    }

def build_trial_df(rows: list[dict]) -> pd.DataFrame:
    ''' Build a df given a list of rows extracted from the datasets'''
    
    for row in rows:
        validate_trial_row(row)

    df = pd.DataFrame(rows)
    
    for col in TRIAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[TRIAL_COLUMNS]
