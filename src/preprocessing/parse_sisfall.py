from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


from common import INTERIM_DIR, RAW_DIR, build_trial_df, make_trial_row

RAW_ROOT = RAW_DIR / "sisfall"
OUT_PATH = INTERIM_DIR / "sisfall" / "SisFall.pkl"
SISFALL_FS_HZ = 200

# ADXL345 (Accelerometer 1)
ADXL345_RESOLUTION = 13
ADXL345_RANGE = 16  # +-16g

# ITG3200 (Gyroscope)
ITG3200_RESOLUTION = 16
ITG3200_RANGE = 2000  # +-2000 deg/s

# MMA8451Q (Accelerometer 2)
MMA8451Q_RESOLUTION = 14
MMA8451Q_RANGE = 8  # +-8g

# Conversion formulas from the SisFall readme
# Acceleration [g]: [(2 * Range) / (2 ^ Resolution)] * raw_bits
# Angular velocity [deg/s]: [(2 * Range) / (2 ^ Resolution)] * raw_bits
ACC_1_CONVERSION_FACTOR = (2 * ADXL345_RANGE) / (2**ADXL345_RESOLUTION)
GYR_CONVERSION_FACTOR = (2 * ITG3200_RANGE) / (2**ITG3200_RESOLUTION)
ACC_2_CONVERSION_FACTOR = (2 * MMA8451Q_RANGE) / (2**MMA8451Q_RESOLUTION)

_NAME_RE = re.compile(r"^(?P<activity>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")

def parse_filename(filename: str) -> dict[str, str | int]:
    match = _NAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Invalid SisFall filename: {filename}")

    activity_id = match.group("activity")
    subject_id = match.group("subject")
    trial_id = match.group("trial")

    return {
        "subject_id": subject_id,
        "trial_id": trial_id,
        "activity_id": activity_id,
        "is_fall": int(activity_id.startswith("F")),
    }

def load_sisfall_acc(path: Path) -> np.ndarray:
    ''' Load the readings only for the first accelerometer '''
    df = pd.read_csv(path, sep=",", header=None, dtype=str, skipinitialspace=True)
    df = df.apply(lambda col: col.str.strip().str.rstrip(";"))
    df = df.apply(pd.to_numeric, errors="raise")

    arr = df.to_numpy(dtype=np.float32)
    if arr.shape[1] < 3:
        raise ValueError(f"SisFall file has less than 3 columns: {path}")

    acc = arr[:, 0:3] * ACC_1_CONVERSION_FACTOR
    return acc.astype(np.float32, copy=False)


def iter_sisfall_rows(raw_root: Path) -> tuple[list[dict], int]:
    ''' Loop through each trial file to create a row'''
    rows: list[dict] = []
    skipped_subject_mismatch = 0

    for file_path in sorted(raw_root.rglob("*.txt")):
        
        if file_path.name.lower() == "readme.txt":
            continue

        meta = parse_filename(file_path.name)
        if file_path.parent.name != meta["subject_id"]:
            skipped_subject_mismatch += 1
            continue

        acc = load_sisfall_acc(file_path)

        rows.append(
            make_trial_row(
                subject_id=str(meta["subject_id"]),
                trial_id=str(meta["trial_id"]),
                activity_id=str(meta["activity_id"]),
                is_fall=int(meta["is_fall"]),
                acc=acc,
                raw_file=file_path,
                sampling_rate_hz=SISFALL_FS_HZ,
            )
        )

    return rows, skipped_subject_mismatch

def main() -> None:
    print("Start parsing raw Sisfall data")
    rows, skipped_subject_mismatch = iter_sisfall_rows(RAW_ROOT)
    df = build_trial_df(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)
    
    print(f"Saved {len(df)} SisFall trials to {OUT_PATH}")
    if skipped_subject_mismatch:
        print(f"Skipped {skipped_subject_mismatch} SisFall files with subject-folder/filename mismatches")

if __name__ == "__main__":
    main()
