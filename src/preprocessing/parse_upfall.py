from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from common import RAW_DIR, INTERIM_DIR, build_trial_df, make_trial_row

RAW_PATH = RAW_DIR / "UP-FALL.csv"
OUT_PATH = INTERIM_DIR / "upfall" / "UP-FALL.pkl"

UPFALL_FS_HZ = 18.0

BeltAccCols = (15, 16, 17)  # 0-based indices; 1-based columns 16-18
SubjectCol = 43
ActivityCol = 44
TrialCol = 45

EXPECTED_HEADER = {
    0: ("TimeStamps", ""),
    15: ("BeltAccelerometer", "x-axis (g)"),
    16: ("", "y-axis (g)"),
    17: ("", "z-axis (g)"),
    43: ("Subject", ""),
    44: ("Activity", ""),
    45: ("Trial", ""),
    46: ("Tag", ""),
}


def _validate_headers(header1: list[str], header2: list[str]) -> None:
    if len(header1) != 47 or len(header2) != 47:
        raise ValueError(f"Unexpected UP-FALL header width: {len(header1)}, {len(header2)}")

    for idx, expected in EXPECTED_HEADER.items():
        pair = (header1[idx], header2[idx])
        if pair != expected:
            raise ValueError(
                f"Unexpected UP-FALL header at column {idx + 1}: expected {expected}, got {pair}"
            )


def _is_fall(activity_id: str) -> int:
    activity_num = int(activity_id)
    if 1 <= activity_num <= 5:
        return 1
    if 6 <= activity_num <= 11:
        return 0
    raise ValueError(f"Unsupported UP-FALL activity ID: {activity_id}")


def load_upfall_trials(path: Path) -> list[dict]:
    grouped_acc: dict[tuple[str, str, str], list[list[float]]] = defaultdict(list)

    with path.open(newline="") as f:
        reader = csv.reader(f)
        
        header1 = next(reader)
        header2 = next(reader)

        _validate_headers(header1, header2)

        for row in reader:
            if not row:
                continue
            if len(row) != 47:
                raise ValueError(f"Unexpected UP-FALL row width {len(row)} in {path}")

            subject_id = row[SubjectCol].strip()
            activity_id = row[ActivityCol].strip()
            trial_id = row[TrialCol].strip()
            key = (subject_id, activity_id, trial_id)

            grouped_acc[key].append(
                [
                    float(row[BeltAccCols[0]]),
                    float(row[BeltAccCols[1]]),
                    float(row[BeltAccCols[2]]),
                ]
            )

    rows: list[dict] = []
    for (subject_id, activity_id, trial_id), samples in sorted(grouped_acc.items()):
        acc = np.asarray(samples, dtype=np.float32)

        rows.append(
            make_trial_row(
                subject_id=subject_id,
                trial_id=trial_id,
                activity_id=activity_id,
                is_fall=_is_fall(activity_id),
                acc=acc,
                raw_file=path,
                sampling_rate_hz=UPFALL_FS_HZ,
            )
        )

    return rows


def main() -> None:

    print("Start parsing raw UMAFall dataset")
    
    df = build_trial_df(load_upfall_trials(RAW_PATH))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)

    print(f"Saved {len(df)} UP-FALL trials to {OUT_PATH}")


if __name__ == "__main__":
    main()
