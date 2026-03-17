from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_DIR = ROOT_DIR / "data" / "raw" / "FallAllD"
INTERIM_DIR = ROOT_DIR / "data" / "interim" / "fallalld"
PICKLE_PATH = INTERIM_DIR / "FallAllD.pkl"

DEVICE_MAP = {
    "D1": "Neck",
    "D2": "Wrist",
    "D3": "Waist",
}


def load_dat_file(path: Path, dtype: np.dtype | None = None) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",")
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    return data


def build_struct(raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw FallAllD directory not found: {raw_dir}")

    acc_files = sorted(raw_dir.glob("*_A.dat"))
    if not acc_files:
        raise FileNotFoundError(f"No accelerometer files found in: {raw_dir}")

    rows: list[dict[str, object]] = []

    for idx, acc_path in enumerate(acc_files, start=1):
        parts = acc_path.stem.split("_")
        if len(parts) != 5:
            raise ValueError(f"Unexpected filename format: {acc_path.name}")

        subject_code, device_code, activity_code, trial_code, sensor_code = parts
        if sensor_code != "A":
            raise ValueError(f"Expected accelerometer seed file, got: {acc_path.name}")

        gyr_path = raw_dir / f"{subject_code}_{device_code}_{activity_code}_{trial_code}_G.dat"
        mag_path = raw_dir / f"{subject_code}_{device_code}_{activity_code}_{trial_code}_M.dat"
        bar_path = raw_dir / f"{subject_code}_{device_code}_{activity_code}_{trial_code}_B.dat"

        missing = [path.name for path in (gyr_path, mag_path, bar_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing paired sensor files for {acc_path.name}: {', '.join(missing)}"
            )

        rows.append(
            {
                "SubjectID": np.uint8(int(subject_code[1:])),
                "Device": DEVICE_MAP.get(device_code, device_code),
                "ActivityID": np.uint8(int(activity_code[1:])),
                "TrialNo": np.uint8(int(trial_code[1:])),
                "Acc": load_dat_file(acc_path, dtype=np.int16),
                "Gyr": load_dat_file(gyr_path, dtype=np.int16),
                "Mag": load_dat_file(mag_path, dtype=np.int16),
                "Bar": load_dat_file(bar_path),
            }
        )

        if idx % 100 == 0 or idx == len(acc_files):
            print(f"Processed file {idx} out of {len(acc_files)}")

    return pd.DataFrame(
        rows,
        columns=["SubjectID", "Device", "ActivityID", "TrialNo", "Acc", "Gyr", "Mag", "Bar"],
    )


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    fallalld = build_struct(RAW_DIR)
    fallalld.to_pickle(PICKLE_PATH)

    print(f"Saved pickle to {PICKLE_PATH}")


if __name__ == "__main__":
    main()
