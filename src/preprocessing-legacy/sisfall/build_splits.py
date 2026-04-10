from __future__ import annotations

from pathlib import Path

import pandas as pd


INDEX_PATH = Path("data/interim/sisfall/index.csv")
OUT_PATH = Path("data/interim/sisfall/splits.csv")


TRAIN_SUBJECTS = {f"SA{idx:02d}" for idx in range(1, 16)}
VAL_SUBJECTS = {f"SA{idx:02d}" for idx in range(16, 18)}

def assign_split(subject: str) -> str:
    if subject in TRAIN_SUBJECTS:
        return "train"
    if subject in VAL_SUBJECTS:
        return "val"
    return "test"


def build_splits(index_path: Path = INDEX_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    df["split"] = df["subject"].map(assign_split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    build_splits()
