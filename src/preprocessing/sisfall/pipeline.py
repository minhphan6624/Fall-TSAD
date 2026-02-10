from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .build_index import build_index
from .build_splits import build_splits
from .filtering import butter_lowpass_filter
from .labeling import find_impact_index, impact_window_labels
from .load_signal import load_signal
from .normalization import apply_zscore, fit_zscore_stats

from .windowing import FS_HZ, OVERLAP, WINDOW_SECONDS, build_window_metadata, segment_signal

RAW_ROOT = Path("data/raw/sisfall")
INTERIM_DIR = Path("data/interim/sisfall")
PROCESSED_DIR = Path("data/processed/sisfall")
INDEX_PATH = INTERIM_DIR / "index.csv"
SPLITS_PATH = INTERIM_DIR / "splits.csv"
WINDOW_META_COLUMNS = [
    "window_local_idx",
    "file_path",
    "subject",
    "activity",
    "trial",
    "is_fall",
    "split",
    "start_idx",
    "end_idx",
    "label_cls",
    "label_tsad",
    "impact_index",
    "tsad_train_eligible",
]

# ----- Helpers to save data -----
def _save_interim_outputs(
    split: str, windows: list[np.ndarray],
    metadata_frames: list[pd.DataFrame], out_dir: Path
):

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
        meta = pd.DataFrame(columns=WINDOW_META_COLUMNS)

    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f"windows_{split}.npz"
    meta_path = out_dir / f"window_meta_{split}.csv"

    np.savez_compressed(x_path, X=x)
    meta.to_csv(meta_path, index=False)

    print(f"Saved interm data for {split} set")
    
    return x, meta

def _save_normalizer(task: str, mean: np.ndarray, std: np.ndarray, out_dir: Path):
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    norm_path = task_dir / "normalizer.npz"

    np.savez_compressed(norm_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved zcore normalizer stats for {task} at {task_dir}")

def _save_final_output( 
    task: str, split: str,
    x: np.ndarray, y: np.ndarray,
    meta: pd.DataFrame, out_dir: Path,
):
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    x_path = task_dir / f"windows_{split}.npz"
    meta_path = task_dir / f"window_meta_{split}.csv"
    
    np.savez_compressed(x_path, X=x.astype(np.float32, copy=False), y=y.astype(np.int8, copy=False))
    meta.to_csv(meta_path, index=False)

    print(f"Saved final processed output for {split} set of {task} task at {task_dir}")


# ----- Main pipeline entrypoint -----
def run_pipeline(
    raw_root: Path = RAW_ROOT,
    out_dir: Path = INTERIM_DIR,
    fs_hz: int = FS_HZ,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = OVERLAP,
):
    
    # Create index + add splits into 
    index_df = build_index(raw_root=raw_root, out_path=INDEX_PATH)
    splits_df = build_splits(index_path=INDEX_PATH, out_path=SPLITS_PATH)

    print(f"Index.csv shapes {index_df.shape}")
    print(f"Splits.csv shapes {splits_df.shape}")

    split_data = {}

    # Create each split
    for split in ("train", "val", "test"):
        
        split_rows = splits_df[splits_df["split"] == split].reset_index(drop=True)
        
        split_windows: list[np.ndarray] = []
        split_meta: list[pd.DataFrame] = []
        file_count = 0
        skipped_short = 0

        # Process each split
        for row in split_rows.itertuples(index=False):
            
            # Extract metadata of that trial file
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
            
            # Labelling process
            window_size = int(round(fs_hz * window_seconds))
            impact_index = find_impact_index(signal) if row.is_fall == 1 else None # Look for impact point if it is a fall signal
            labels = impact_window_labels(
                start_indices=start_indices, window_size=window_size,
                n_samples=signal.shape[0], is_fall=bool(row.is_fall),
                impact_index=impact_index, fs_hz=fs_hz,
            )

            # Build metadata for a single window
            meta_df = build_window_metadata(
                start_indices=start_indices, file_meta=file_meta,
                fs_hz=fs_hz, window_seconds=window_seconds,
                split=row.split, file_path=str(file_path),
            )

            meta_df["label_cls"] = labels.astype(np.int8)
            meta_df["label_tsad"] = labels.astype(np.int8)
            meta_df["impact_index"] = -1 if impact_index is None else int(impact_index)
            meta_df["tsad_train_eligible"] = np.int8(1 if (row.split == "train" and row.is_fall == 0) else 0)

            split_windows.append(windows)
            split_meta.append(meta_df)
            file_count += 1

        split_x, split_meta_df = _save_interim_outputs( split, split_windows, split_meta, out_dir)
        split_data[split] = {"x": split_x, "meta": split_meta_df}
        
        print(f"Processed {split} set: {file_count} files, skipped: {skipped_short}, num_windows: {split_x.shape[0]}")

    # ----- Normalization -----

    train_x = split_data["train"]["x"]  
    train_meta = split_data["train"]["meta"]  
    
    assert isinstance(train_x, np.ndarray)
    assert isinstance(train_meta, pd.DataFrame)

    # fit z-score stats on training 
    cls_mean, cls_std = fit_zscore_stats(train_x)
    _save_normalizer("classification", cls_mean, cls_std, PROCESSED_DIR)

    # Fit Z-score stats for TSAD: For training ADLs only
    tsad_train_x = train_x[train_meta["tsad_train_eligible"].to_numpy(dtype=np.int8) == 1]
    tsad_mean, tsad_std = fit_zscore_stats(tsad_train_x)
    _save_normalizer("tsad", tsad_mean, tsad_std, PROCESSED_DIR)

    # Apply fitted statistics accordingly
    for split in ("train", "val", "test"):
        x = split_data[split]["x"]  
        meta = split_data[split]["meta"]  
        assert isinstance(x, np.ndarray)
        assert isinstance(meta, pd.DataFrame)

        # Apply z-score normalization on classification set
        x_cls = apply_zscore(x, cls_mean, cls_std)
        y_cls = meta["label_cls"].to_numpy(dtype=np.int8)
        _save_final_output("classification", split, x_cls, y_cls, meta, PROCESSED_DIR)

        # Apply z-socre normalization on tsad set
        x_tsad = apply_zscore(x, tsad_mean, tsad_std)
        y_tsad = meta["label_tsad"].to_numpy(dtype=np.int8)
        _save_final_output("tsad", split, x_tsad, y_tsad, meta, PROCESSED_DIR)

        print(f"Classifications total windows for {split} set: {x_cls.shape[0]}")
        print(f"TSAD total windows for {split} set: {x_tsad.shape[0]}")

if __name__ == "__main__":
    run_pipeline()
