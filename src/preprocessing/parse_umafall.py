from __future__ import annotations

from pathlib import Path
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
    raw_trial = str(int(match.group("trial")))
    date = match.group("date")
    time = match.group("time")

    return {
        "subject_id": f"Subject_{match.group('subject')}",
        "trial_id": f"{raw_trial}_{date}_{time}",
        "activity_id": match.group("activity"),
        "is_fall": int(kind == "Fall"),
    }

def load_umafall_file(path: Path) -> np.ndarray | None:
    samples: list[list[float]] = []

    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            parts = [value.strip() for value in line.split(";")]
            if len(parts) != 7:
                raise ValueError(f"Unexpected UMAFall row width {len(parts)} in {path}")

            sensor_type = int(parts[5])
            sensor_id = int(parts[6])
            if sensor_type != ACC_SENSOR_TYPE or sensor_id != WAIST_SENSOR_ID:
                continue

            samples.append([float(parts[2]), float(parts[3]), float(parts[4])])

    if not samples:
        return None

    return np.asarray(samples, dtype=np.float32)

def iter_umafall_rows(raw_root: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped_files = 0

    for file_path in sorted(raw_root.glob("*.csv")):
        meta = parse_filename(file_path.name)
        acc = load_umafall_file(file_path)

        if acc is None:
            skipped_files += 1
            continue

        rows.append(
            make_trial_row(
                dataset=DATASET_NAME,
                subject_id=str(meta["subject_id"]),
                trial_id=str(meta["trial_id"]),
                activity_id=str(meta["activity_id"]),
                is_fall=int(meta["is_fall"]),
                acc=acc,
                raw_file=file_path,
                sampling_rate_hz=20.0,
            )
        )

    return rows, skipped_files

def main() -> None:
    rows, skipped_files = iter_umafall_rows(RAW_ROOT)
    df = build_trial_df(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)

    print(f"Saved {len(df)} UMAFall trials to {OUT_PATH}")
    print(f"Skipped {skipped_files} UMAFall files without waist accelerometer data")


if __name__ == "__main__":
    main()
