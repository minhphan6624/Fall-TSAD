from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .parse_filename import parse_filename


RAW_ROOT = Path("data/raw/sisfall")
OUT_PATH = Path("data/interim/sisfall_v2/index.csv")


def _iter_data_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.txt"):
        if path.name.lower() == "readme.txt":
            continue
        yield path


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def build_index(raw_root: Path = RAW_ROOT, out_path: Path = OUT_PATH) -> pd.DataFrame:
    records = []

    for file_path in _iter_data_files(raw_root):
        try:
            meta = parse_filename(file_path.name)
        except ValueError:
            continue
        
        meta.update(
            {
                "path": str(file_path),
                "n_samples": _count_lines(file_path),
            }
        )
        
        records.append(meta)

    if not records:
        raise RuntimeError(f"No valid data files found under {raw_root}")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["subject", "activity", "trial"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    return df


if __name__ == "__main__":
    build_index()
