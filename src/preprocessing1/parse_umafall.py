from __future__ import annotations

from pathlib import Path
import csv
import re

import numpy as np

from common import RAW_DIR, INTERIM_DIR, build_trial_df, make_trial_row

DATASET_NAME = "umafall"
RAW_ROOT = RAW_DIR / "UMAFall"
OUT_PATH = INTERIM_DIR / "umafall" / "UMAFall.pkl"


WAIST_SENSOR_ID = 2
ACC_SENSOR_TYPE = 0

_NAME_RE = re.compile(
    r"^UMAFall_Subject_(?P<subject>\d{2})_(?P<kind>ADL|Fall)_(?P<activity>.+)_(?P<trial>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.csv$"
)

def parse_filename(filename: str) -> dict[str, str | int]:
    match = _NAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Invalid UMAFall filename: {filename}")

    kind = match.group("kind")
    
    return {
        "subject_id": f"Subject_{match.group('subject')}",
        "trial_id": str(int(match.group("trial"))),
        "activity_id": match.group("activity"),
        "is_fall": int(kind == "Fall"),
    }

def load_umafall_file(path: Path):
    pass


def iter_umafall_rows(raw_root: Path):
    pass

def main() -> None:
    df = build_trial_df(iter_umafall_rows(RAW_ROOT))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)

    print(f"Saved {len(df)} UMAFall trials to {OUT_PATH}")


if __name__ == "__main__":
    main()
