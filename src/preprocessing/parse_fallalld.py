from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from common import INTERIM_DIR, RAW_DIR, build_trial_df, make_trial_row

RAW_ROOT = RAW_DIR / "FallAllD"
OUT_PATH = INTERIM_DIR / "fallalld" / "FallAllD.pkl"

# Raw values are preserved intentionally for now. The documented ±8g range and
# sensor model are not sufficient on their own for a defensible unit conversion.
FALLALLD_FS_HZ = 238

WAIST_DEVICE_CODE = "D3"
ADL_ACTIVITY_IDS = {f"A{i:03d}" for i in range(1, 45)}
FALL_ACTIVITY_IDS = {f"A{i:03d}" for i in range(101, 136)}
VALID_ACTIVITY_IDS = ADL_ACTIVITY_IDS | FALL_ACTIVITY_IDS

_NAME_RE = re.compile(
    r"^(?P<subject>S\d{2})_(?P<device>D\d)_(?P<activity>A\d{3})_(?P<trial>T\d{2})_A\.dat$"
)


def parse_filename(filename: str) -> dict[str, str | int]:
    match = _NAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Invalid FallAllD filename: {filename}")

    subject_id = match.group("subject")
    device_code = match.group("device")
    activity_id = match.group("activity")
    trial_id = match.group("trial")

    if device_code != WAIST_DEVICE_CODE:
        raise ValueError(f"Expected waist device {WAIST_DEVICE_CODE}, got {device_code} in {filename}")

    if activity_id not in VALID_ACTIVITY_IDS:
        raise ValueError(f"Unsupported FallAllD activity ID: {activity_id}")

    return {
        "subject_id": subject_id,
        "trial_id": trial_id,
        "activity_id": activity_id,
        "is_fall": int(activity_id in FALL_ACTIVITY_IDS),
    }

def iter_fallalld_rows(raw_root: Path) -> list[dict]:
    ''' Loop through each trial file to create a row'''
    rows: list[dict] = []

    for file_path in sorted(raw_root.glob(f"*_{WAIST_DEVICE_CODE}_*_A.dat")):
        meta = parse_filename(file_path.name)
        acc = np.genfromtxt(file_path, delimiter=',').astype(np.float32, copy=False)

        rows.append(
            make_trial_row(
                subject_id=str(meta["subject_id"]),
                trial_id=str(meta["trial_id"]),
                activity_id=str(meta["activity_id"]),
                is_fall=int(meta["is_fall"]),
                acc=acc,
                raw_file=file_path,
                sampling_rate_hz=FALLALLD_FS_HZ,
            )
        )

    return rows

def main() -> None:

    df = build_trial_df(iter_fallalld_rows(RAW_ROOT))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PATH)
    
    print(f"Saved {len(df)} FallAllD trials to {OUT_PATH}")

if __name__ == "__main__":
    main()
