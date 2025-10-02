from pathlib import Path
import pandas as pd
from .build_metadata import build_metadata
from .split import split_data_by_subject as split_subjects
from .normalize import normalize_splits
from .serialize import serialize

def run_pipeline(cfg):
    raw_dir = cfg.data.raw_dir
    out_dir = Path(cfg.data.proc_dir)
    seed = cfg.seed

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    metadata_df = build_metadata(raw_dir)
    metadata_df.to_csv(out_dir / "metadata.csv", index=False)

    # 2. Splits
    splits = split_subjects(metadata_df, cfg.data.split, seed)

    # 3–4. Normalize
    normed = normalize_splits(splits)

    # 5–6. Segment + serialize
    serialize(normed, out_dir, 
              window=int(cfg.data.segment_seconds*cfg.data.sampling_rate),
              overlap=cfg.data.overlap)

