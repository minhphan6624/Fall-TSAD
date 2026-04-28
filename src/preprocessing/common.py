from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TRIAL_COLUMNS = [ "subject_id", "trial_id", "activity_id", "is_fall",
                 "sampling_rate_hz", "n_samples", "acc", "raw_file",]

# Create a row for a trial
def make_trial_row( subject_id: str, trial_id: str, activity_id: str,
    is_fall: int | bool, acc: np.ndarray, raw_file: str | Path, 
    sampling_rate_hz: int | float,
) -> dict:
    return {
        "subject_id": subject_id,
        "trial_id": trial_id,
        "activity_id": activity_id,
        "is_fall": int(is_fall),
        "sampling_rate_hz": float(sampling_rate_hz),
        "n_samples": int(acc.shape[0]),
        "acc": acc,
        "raw_file": str(raw_file),
    }

def build_trial_df(rows: list[dict]):
    ''' Build a df given a list of rows extracted from the datasets'''

    df = pd.DataFrame(rows)
    
    for col in TRIAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[TRIAL_COLUMNS]
