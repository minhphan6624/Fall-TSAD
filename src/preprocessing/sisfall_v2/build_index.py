from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .parse_filename import parse_filename


RAW_ROOT = Path("data/raw/sisfall")
OUT_PATH = Path("data/interim/sisfall/index.csv")

_NAME_RE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")

def parse_filename(filename: str):
    ''' Parse the name of a trial file'''
    m = _NAME_RE.match(filename)
    if not m:
        raise ValueError(f"Invalid filename {filename}")
    
    # Extract metadata from filename
    activity = m.group("code")
    subject = m.group("subject")
    group = "adult" if subject.startswith("SA") else "elderly"
    trial = m.group("trial")
    is_fall = activity.startswith("F")
    
    return {
        "activity": activity,
        "subject": subject,
        "group": group,
        "trial": int(trial[1:]),
        "is_fall": int(is_fall),
    }

def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def build_index(raw_root: Path = RAW_ROOT, out_path: Path = OUT_PATH) -> pd.DataFrame:
    records = []

    for file_path in raw_root.rglob("*.txt"):
        if file_path.name.lower() == "readme.txt":
            continue

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

    # Create df and sort
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["subject", "activity", "trial"]).reset_index(drop=True)

    # Save df to csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    return df


if __name__ == "__main__":
    build_index()
