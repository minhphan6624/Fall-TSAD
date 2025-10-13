from pathlib import Path
from .build_metadata import build_metadata
from .split import split_data_custom as split_data
from .normalize import normalize_splits
from .serialize import serialize

def run_pipeline(cfg):
    raw_dir = Path(cfg.data.raw_dir)
    out_dir = Path(cfg.data.processed_dir)
    seed = cfg.seed

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    metadata_df = build_metadata(raw_dir)
    metadata_df.to_csv(out_dir / "metadata.csv", index=False)

    # 2. Splits
    splits = split_data(metadata_df, cfg.data.split.val_size, seed)
    
    # Save split metadata to CSV files as per DATA_SPLITTING_GUIDE.md
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        split_df.to_csv(splits_dir / f"{split_name}.csv", index=False)

    # 3–4. Normalize
    normed = normalize_splits(splits)
    
    # 5. Windowing
    # Calculate window size and overlap in time steps
    # Example: windows size: 3seconds * 200Hz = 600 time steps
    # Overlap: 50% of 600 = 300 time steps
    window_size = int(cfg.data.segment_seconds * cfg.data.sampling_rate)
    if cfg.data.overlap >= window_size:
        raise ValueError("Overlap must be less than window size.")
    
    overlap = int(cfg.data.overlap * window_size) if cfg.data.overlap < 1 else int(cfg.data.overlap)

    print(f"Window size (in time steps): {window_size}")
    print(f"Overlap (in time steps): {overlap}")

    # 5–6. Segment + serialize
    serialize(normed, out_dir, window=window_size, overlap=overlap, sampling_rate=cfg.data.sampling_rate)
