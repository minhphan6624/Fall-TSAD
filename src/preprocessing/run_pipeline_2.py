import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.build_splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_SEED,
    DEFAULT_SPLIT_PROTOCOL,
    KFOLD_SPLIT_PROTOCOL,
    build_split_artifacts,
    save_split_artifacts,
)
from src.preprocessing.common import INTERIM_DIR, PROCESSED_DIR
from src.preprocessing.label_windows import label_windows
from src.preprocessing.normalize_windows import apply_zscore, fit_zscore_stats
from src.preprocessing.resampling import resample_trials_df
from src.preprocessing.window_trials import DEFAULT_OVERLAP, DEFAULT_WINDOW_SECONDS, generate_windows

INTERIM_PICKLE_NAMES = {
    "sisfall": "SisFall.pkl",
    "fallalld": "FallAllD.pkl",
    "umafall": "UMAFall.pkl",
    "upfall": "UP-FALL.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline for a dataset.")
    
    parser.add_argument(
        "--dataset", required=True,
        choices=sorted(INTERIM_PICKLE_NAMES.keys()),
        help="Dataset name to process using the repo's standard directory layout.",
    )
    parser.add_argument(
        "--seed", type=int,
        default=DEFAULT_SEED,
        help="Random seed used by the subject split builder.",
    )
    parser.add_argument(
        "--manual-split-csv",
        type=Path, default=None,
        help="Optional manual split CSV. Omit to use automatic subject-wise splits.",
    )
    parser.add_argument(
        "--split-protocol",
        choices=(DEFAULT_SPLIT_PROTOCOL, KFOLD_SPLIT_PROTOCOL),
        default=DEFAULT_SPLIT_PROTOCOL,
        help="Subject split protocol. Manual split CSV overrides this setting.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of subject folds for subject_kfold.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Fold index to use as the test split for subject_kfold.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help="Fractional window overlap in [0, 1).",
    )
    parser.add_argument(
        "--target-sampling-rate-hz",
        type=float,
        default=None,
        help="Optional target sampling rate. If provided, trial acc arrays are resampled before windowing.",
    )
    parser.add_argument(
        "--allow-upsample",
        action="store_true",
        help="Allow upsampling when the target sampling rate is higher than a trial's source rate.",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Optional processed output dataset name. Useful for variants such as sisfall_20hz.",
    )
    parser.add_argument(
        "--adl-filter-config",
        type=Path,
        default=None,
        help="Optional JSON mapping dataset names to normal training activity IDs to keep.",
    )
    return parser.parse_args()


def export_mode_split( out_dir: Path, mode: str, split_name: str, 
    X: np.ndarray, metadata_df: pd.DataFrame,
    adl_activity_ids: list[str] | None = None):
    ''' Export the final output of the preprocessing pipeline based on the learning mode '''

    split_meta = metadata_df[metadata_df["split"] == split_name].copy()
    if split_name == "train":
        if mode == "tsad":
            split_meta = split_meta[split_meta["window_label"] == 0].copy()
        if adl_activity_ids is not None:
            keep = split_meta["activity_id"].isin(adl_activity_ids)
            if mode == "classification":
                keep = keep | (split_meta["window_label"] == 1)
            split_meta = split_meta[keep].copy()

    split_meta = split_meta.reset_index(drop=True)
    indices = split_meta["window_id"].to_numpy(dtype=np.int64)
    
    X_split = X[indices]
    y_split = split_meta["window_label"].to_numpy(dtype=np.int64)

    np.savez_compressed(out_dir / f"windows_{split_name}.npz", X=X_split, y=y_split)
    
    split_meta.to_csv(out_dir / f"window_meta_{split_name}.csv", index=False)


def run_pipeline(
    dataset: str, seed: int = DEFAULT_SEED,
    manual_split_csv: Path | None = None,
    split_protocol: str = DEFAULT_SPLIT_PROTOCOL,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_index: int = 0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP,
    target_sampling_rate_hz: float | None = None,
    allow_upsample: bool = False,
    output_dataset: str | None = None,
    adl_filter_config: Path | None = None,
) -> Path:
    interim_path = INTERIM_DIR / dataset / INTERIM_PICKLE_NAMES[dataset]
    dataset_dir = PROCESSED_DIR / (output_dataset or dataset)
    adl_activity_ids = None
    if adl_filter_config is not None:
        with adl_filter_config.open() as f:
            adl_activity_ids = json.load(f)[dataset]

    # ----- Step 1: Load trial-level interim data -----
    trials_df = pd.read_pickle(interim_path)

    # ----- Step 2: Optional trial-level resampling -----
    if target_sampling_rate_hz is not None:
        trials_df = resample_trials_df(
            trials_df=trials_df,
            target_hz=target_sampling_rate_hz,
            allow_upsample=allow_upsample,
        )

    # ----- Step 3: Subject Splitting -----
    subject_summary, subject_splits, trials_with_split = build_split_artifacts(
        trials_df=trials_df,
        seed=seed,
        manual_split_csv=manual_split_csv,
        split_protocol=split_protocol,
        n_folds=n_folds,
        fold_index=fold_index,
    )
    save_split_artifacts(subject_summary, subject_splits, trials_with_split, dataset_dir)
    if adl_activity_ids is not None:
        with (dataset_dir / "adl_filter.json").open("w") as f:
            json.dump(
                {"dataset": dataset, "normal_train_activity_ids": adl_activity_ids},
                f,
                indent=2,
            )
            f.write("\n")

    # ----- Step 4: Windowing -----
    X_raw, window_meta = generate_windows(
        trials_df=trials_with_split,
        window_seconds=window_seconds,
        overlap=overlap,
    )

    # ----- Step 5: Labelling -----
    labeled_meta = label_windows(trials_df=trials_with_split, 
                                 window_meta_df=window_meta)
    

    # ----- Prelim: Save raw windows before norm ----
    raw_dir = dataset_dir / "raw_windows"
    raw_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(raw_dir / "windows_all.npz", X=X_raw)
    labeled_meta.to_csv(raw_dir / "window_meta_all.csv", index=False)

    # ----- Step 6: Preform training mode-based normalization -----
    for mode in ("classification", "tsad"):

        mean, std = fit_zscore_stats(X_raw, labeled_meta, mode=mode)
        X_norm = apply_zscore(X_raw, mean, std)

        mode_dir = dataset_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # Save normalizer
        np.savez_compressed(
            mode_dir / "normalizer.npz",
            mean=mean, std=std,
            mode=np.asarray(mode),
        )

        for split_name in ("train", "val", "test"):
            export_mode_split(
                out_dir=mode_dir,
                split_name=split_name,
                X=X_norm,
                metadata_df=labeled_meta,
                mode=mode,
                adl_activity_ids=adl_activity_ids,
            )

    return dataset_dir


def main() -> None:
    args = parse_args()
    
    out_dir = run_pipeline(
        dataset=args.dataset,
        seed=args.seed,
        manual_split_csv=args.manual_split_csv,
        split_protocol=args.split_protocol,
        n_folds=args.n_folds,
        fold_index=args.fold_index,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        target_sampling_rate_hz=args.target_sampling_rate_hz,
        allow_upsample=args.allow_upsample,
        output_dataset=args.output_dataset,
        adl_filter_config=args.adl_filter_config,
    )

    print(f"Saved processed artifacts for {args.dataset} to {out_dir}")


if __name__ == "__main__":
    main()
