from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .build_index import RAW_ROOT, build_index
from .build_splits import build_splits
from .filtering import butter_lowpass_filter
from .load_signal import load_signal

from .windowing import FS_HZ, OVERLAP, WINDOW_SECONDS, build_window_metadata, segment_signal

INTERIM_DIR = Path("data/interim/sisfall")
INDEX_PATH = INTERIM_DIR / "index.csv"
SPLITS_PATH = INTERIM_DIR / "splits.csv"


def _save_split_outputs(
    split: str, windows: list[np.ndarray],
    metadata_frames: list[pd.DataFrame], out_dir: Path,
) -> tuple[Path, Path]:

    # Concatenate all the window df for a split
    if windows:
        x = np.concatenate(windows, axis=0).astype(np.float32, copy=False)
    else:
        window_size = int(round(FS_HZ * WINDOW_SECONDS))
        x = np.empty((0, window_size, 3), dtype=np.float32)

    # Concatenate all the metadata df for a split
    if metadata_frames:
        meta = pd.concat(metadata_frames, ignore_index=True)
    else:
        meta = pd.DataFrame(
            columns=[
                "window_local_idx",
                "file_path",
                "subject",
                "activity",
                "trial",
                "is_fall",
                "split",
                "start_idx",
                "end_idx",
            ]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f"windows_{split}.npz"
    meta_path = out_dir / f"window_meta_{split}.csv"

    np.savez_compressed(x_path, X=x)
    meta.to_csv(meta_path, index=False)
    
    return x_path, meta_path


def run_pipeline(
    raw_root: Path = RAW_ROOT,
    out_dir: Path = INTERIM_DIR,
    fs_hz: int = FS_HZ,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = OVERLAP,
) -> dict[str, dict[str, int | str]]:
    
    # Create index + add splits into 
    index_df = build_index(raw_root=raw_root, out_path=INDEX_PATH)
    splits_df = build_splits(index_path=INDEX_PATH, out_path=SPLITS_PATH)

    summary: dict[str, dict[str, int | str]] = {}

    for split in ("train", "val", "test"):
        
        split_rows = splits_df[splits_df["split"] == split].reset_index(drop=True)
        
        split_windows: list[np.ndarray] = []
        split_meta: list[pd.DataFrame] = []
        file_count = 0
        skipped_short = 0

        for row in split_rows.itertuples(index=False):
            file_path = Path(row.path)

            file_meta = {
                "subject": row.subject,
                "activity": row.activity,
                "trial": row.trial,
                "is_fall": row.is_fall,
                "split": row.split,
            }
            
            # Convert + butterworth filter
            signal = load_signal(file_path)["acc1"]
            signal = butter_lowpass_filter(signal, fs_hz=fs_hz, cutoff_hz=5.0, order=4)
            
            # Segment signal into windows
            windows, start_indices = segment_signal(
                signal=signal,
                fs_hz=fs_hz,
                window_seconds=window_seconds,
                overlap=overlap,
            )

            if windows.shape[0] == 0:
                skipped_short += 1
                continue

            meta_df = build_window_metadata(
                start_indices=start_indices,
                file_meta=file_meta,
                fs_hz=fs_hz,
                window_seconds=window_seconds,
                split=row.split,
                file_path=str(file_path),
            )

            split_windows.append(windows)
            split_meta.append(meta_df)
            file_count += 1

        x_path, meta_path = _save_split_outputs(split, split_windows, split_meta, out_dir)
        
        # Summary object for printing out results once finished
        summary[split] = {
            "files_processed": file_count,
            "files_skipped_short": skipped_short,
            "num_windows": int(sum(w.shape[0] for w in split_windows)),
            "windows_path": str(x_path),
            "metadata_path": str(meta_path),
        }

    summary["index_rows"] = {"count": int(index_df.shape[0]), "path": str(INDEX_PATH)}
    summary["splits_rows"] = {"count": int(splits_df.shape[0]), "path": str(SPLITS_PATH)}
    return summary


if __name__ == "__main__":
    result = run_pipeline()
    for key, value in result.items():
        print(f"{key}: {value}")
