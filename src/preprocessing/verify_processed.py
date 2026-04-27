from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.common import INTERIM_PICKLE_NAMES, PROCESSED_DIR


DATASETS = tuple(sorted(INTERIM_PICKLE_NAMES))
SPLITS = ("train", "val", "test")
MODES = ("classification", "tsad")
RAW_META_COLUMNS = [
    "window_id",
    "subject_id",
    "activity_id",
    "trial_id",
    "is_fall",
    "split",
    "sampling_rate_hz",
    "start_idx",
    "end_idx",
    "window_label",
    "tsad_train_eligible",
]


@dataclass
class CheckResult:
    dataset: str
    errors: list[str]
    warnings: list[str]
    summary: dict[str, object]


def _read_npz_x(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "X" not in data:
            raise ValueError(f"{path} does not contain an X array")
        return data["X"]


def _read_npz_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        missing = {"X", "y"} - set(data.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return data["X"], data["y"]


def _require_files(paths: list[Path], errors: list[str]) -> None:
    for path in paths:
        if not path.exists():
            errors.append(f"missing required file: {path}")


def _check_subject_disjoint(meta: pd.DataFrame, errors: list[str]) -> None:
    split_subjects = {
        split: set(meta.loc[meta["split"] == split, "subject_id"].astype(str))
        for split in SPLITS
    }
    for left_idx, left in enumerate(SPLITS):
        for right in SPLITS[left_idx + 1 :]:
            overlap = split_subjects[left] & split_subjects[right]
            if overlap:
                errors.append(
                    f"subject leakage between {left} and {right}: "
                    f"{sorted(overlap)[:10]}"
                )


def _check_raw_windows(dataset_dir: Path, dataset: str) -> tuple[pd.DataFrame | None, dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, object] = {}
    raw_npz = dataset_dir / "raw_windows" / "windows_all.npz"
    raw_meta_csv = dataset_dir / "raw_windows" / "window_meta_all.csv"
    _require_files([raw_npz, raw_meta_csv], errors)
    if errors:
        return None, summary, errors, warnings

    X = _read_npz_x(raw_npz)
    meta = pd.read_csv(raw_meta_csv)

    missing_cols = [col for col in RAW_META_COLUMNS if col not in meta.columns]
    if missing_cols:
        errors.append(f"raw metadata missing columns: {missing_cols}")

    if X.ndim != 3 or X.shape[2] != 3:
        errors.append(f"raw X must have shape (N, T, 3), got {X.shape}")
    if len(meta) != X.shape[0]:
        errors.append(f"raw X/meta row mismatch: X has {X.shape[0]}, meta has {len(meta)}")
    if not np.isfinite(X).all():
        errors.append("raw X contains NaN or inf")

    if "window_id" in meta:
        expected = np.arange(len(meta), dtype=np.int64)
        actual = meta["window_id"].to_numpy(dtype=np.int64, copy=False)
        if len(actual) != len(np.unique(actual)):
            errors.append("raw metadata has duplicate window_id values")
        if len(actual) == len(expected) and not np.array_equal(actual, expected):
            errors.append("raw metadata window_id is not contiguous from 0 to N-1")

    if {"start_idx", "end_idx"}.issubset(meta.columns):
        starts = meta["start_idx"].to_numpy()
        ends = meta["end_idx"].to_numpy()
        if np.any(ends <= starts):
            errors.append("raw metadata has windows with end_idx <= start_idx")
        if X.ndim == 3 and np.any((ends - starts) != X.shape[1]):
            errors.append("metadata window lengths do not match raw X time dimension")

    if "split" in meta:
        found_splits = set(meta["split"].dropna().astype(str))
        unexpected = found_splits - set(SPLITS)
        missing = set(SPLITS) - found_splits
        if unexpected:
            errors.append(f"unexpected split values: {sorted(unexpected)}")
        if missing:
            errors.append(f"missing split values in windows: {sorted(missing)}")
        _check_subject_disjoint(meta, errors)

    if {"is_fall", "window_label", "tsad_train_eligible"}.issubset(meta.columns):
        for col in ("is_fall", "window_label", "tsad_train_eligible"):
            values = set(meta[col].dropna().astype(int).unique())
            if values - {0, 1}:
                errors.append(f"{col} must be binary, got {sorted(values)}")
        false_fall_windows = meta[(meta["is_fall"].astype(int) == 0) & (meta["window_label"].astype(int) == 1)]
        if len(false_fall_windows):
            errors.append(f"{len(false_fall_windows)} ADL windows are labeled as falls")
        bad_eligible = meta[
            (meta["tsad_train_eligible"].astype(int) == 1)
            & ((meta["split"] != "train") | (meta["window_label"].astype(int) != 0))
        ]
        if len(bad_eligible):
            errors.append(f"{len(bad_eligible)} invalid tsad_train_eligible rows")

        fall_trials = meta.loc[meta["is_fall"].astype(int) == 1, ["subject_id", "activity_id", "trial_id"]].drop_duplicates()
        labeled_fall_trials = meta.loc[meta["window_label"].astype(int) == 1, ["subject_id", "activity_id", "trial_id"]].drop_duplicates()
        missing_labeled = len(fall_trials.merge(labeled_fall_trials, how="left", indicator=True).query("_merge == 'left_only'"))
        if missing_labeled:
            warnings.append(f"{missing_labeled} fall trials produced no positive windows")

    if {"split", "window_label"}.issubset(meta.columns):
        split_counts = meta.groupby(["split", "window_label"]).size().unstack(fill_value=0)
        summary["raw_split_label_counts"] = split_counts.to_dict()
    summary["raw_shape"] = tuple(int(v) for v in X.shape)
    summary["n_subjects"] = int(meta["subject_id"].nunique()) if "subject_id" in meta else None
    summary["dataset"] = dataset
    return meta, summary, errors, warnings


def _check_normalizer(mode_dir: Path, X_raw: np.ndarray, meta: pd.DataFrame, mode: str, errors: list[str]) -> None:
    normalizer_path = mode_dir / "normalizer.npz"
    with np.load(normalizer_path) as data:
        for key in ("mean", "std", "mode"):
            if key not in data:
                errors.append(f"{normalizer_path} missing {key}")
                return
        mean = data["mean"]
        std = data["std"]
        saved_mode = str(data["mode"])

    if saved_mode != mode:
        errors.append(f"{normalizer_path} stores mode={saved_mode!r}, expected {mode!r}")
    if mean.shape != (3,) or std.shape != (3,):
        errors.append(f"{normalizer_path} mean/std must have shape (3,), got {mean.shape}/{std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        errors.append(f"{normalizer_path} mean/std contains NaN or inf")
    if np.any(std <= 0):
        errors.append(f"{normalizer_path} std must be positive")

    mask = meta["split"].eq("train").to_numpy(copy=True)
    if mode == "tsad":
        mask &= meta["window_label"].eq(0).to_numpy()
    expected_mean = X_raw[mask].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_std = X_raw[mask].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_std = np.where(expected_std == 0.0, 1.0, expected_std).astype(np.float32)
    if not np.allclose(mean, expected_mean, rtol=1e-5, atol=1e-5):
        errors.append(f"{mode} normalizer mean does not match raw training windows")
    if not np.allclose(std, expected_std, rtol=1e-5, atol=1e-5):
        errors.append(f"{mode} normalizer std does not match raw training windows")


def _check_mode_exports(dataset_dir: Path, meta_all: pd.DataFrame, errors: list[str]) -> dict[str, object]:
    raw_X = _read_npz_x(dataset_dir / "raw_windows" / "windows_all.npz")
    summary: dict[str, object] = {}
    for mode in MODES:
        mode_dir = dataset_dir / mode
        required = [mode_dir / "normalizer.npz"]
        for split in SPLITS:
            required.extend(
                [
                    mode_dir / f"windows_{split}.npz",
                    mode_dir / f"window_meta_{split}.csv",
                ]
            )
        _require_files(required, errors)
        if any(not path.exists() for path in required):
            continue

        _check_normalizer(mode_dir, raw_X, meta_all, mode, errors)
        with np.load(mode_dir / "normalizer.npz") as normalizer:
            saved_mean = normalizer["mean"].reshape(1, 1, -1)
            saved_std = normalizer["std"].reshape(1, 1, -1)
        mode_counts = {}
        for split in SPLITS:
            split_npz = mode_dir / f"windows_{split}.npz"
            split_meta_csv = mode_dir / f"window_meta_{split}.csv"
            X, y = _read_npz_xy(split_npz)
            split_meta = pd.read_csv(split_meta_csv)
            mode_counts[split] = int(len(split_meta))

            if X.ndim != 3 or X.shape[2] != 3:
                errors.append(f"{split_npz} must have shape (N, T, 3), got {X.shape}")
            if len(split_meta) != X.shape[0] or len(y) != X.shape[0]:
                errors.append(f"{mode}/{split} X/y/meta counts do not match")
            if not np.isfinite(X).all():
                errors.append(f"{split_npz} contains NaN or inf")
            if "split" in split_meta and not split_meta["split"].eq(split).all():
                errors.append(f"{mode}/{split} metadata contains rows from other splits")
            if "window_label" in split_meta and not np.array_equal(y, split_meta["window_label"].to_numpy(dtype=np.int64)):
                errors.append(f"{mode}/{split} y does not match metadata window_label")
            if mode == "tsad" and split == "train":
                if len(y) and np.any(y != 0):
                    errors.append("tsad/train contains positive labels")
                if "tsad_train_eligible" in split_meta and not split_meta["tsad_train_eligible"].eq(1).all():
                    errors.append("tsad/train metadata contains ineligible rows")

            raw_ids = set(meta_all["window_id"].astype(int))
            split_ids = set(split_meta["window_id"].astype(int)) if "window_id" in split_meta else set()
            if not split_ids <= raw_ids:
                errors.append(f"{mode}/{split} metadata references unknown window_id values")
            elif len(split_meta):
                indices = split_meta["window_id"].to_numpy(dtype=np.int64)
                expected_X = ((raw_X[indices] - saved_mean) / saved_std).astype(np.float32)
                if not np.allclose(X, expected_X, rtol=1e-6, atol=1e-6):
                    max_abs = float(np.max(np.abs(X - expected_X)))
                    errors.append(
                        f"{mode}/{split} X does not match raw X transformed by saved normalizer "
                        f"(max_abs_diff={max_abs:.6g})"
                    )

        summary[f"{mode}_counts"] = mode_counts
    return summary


def verify_dataset(dataset: str) -> CheckResult:
    dataset_dir = PROCESSED_DIR / dataset
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, object] = {}

    _require_files(
        [
            dataset_dir / "subject_summary.csv",
            dataset_dir / "subject_splits.csv",
            dataset_dir / "trials_with_split.pkl",
        ],
        errors,
    )

    meta_all, raw_summary, raw_errors, raw_warnings = _check_raw_windows(dataset_dir, dataset)
    summary.update(raw_summary)
    errors.extend(raw_errors)
    warnings.extend(raw_warnings)

    if meta_all is not None:
        summary.update(_check_mode_exports(dataset_dir, meta_all, errors))

    return CheckResult(dataset=dataset, errors=errors, warnings=warnings, summary=summary)


def print_result(result: CheckResult) -> None:
    status = "PASS" if not result.errors else "FAIL"
    raw_shape = result.summary.get("raw_shape", "?")
    n_subjects = result.summary.get("n_subjects", "?")
    print(f"\n[{status}] {result.dataset}: raw_shape={raw_shape}, subjects={n_subjects}")

    for key in ("classification_counts", "tsad_counts"):
        if key in result.summary:
            print(f"  {key}: {result.summary[key]}")
    if "raw_split_label_counts" in result.summary:
        print(f"  raw_split_label_counts: {result.summary['raw_split_label_counts']}")

    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    for error in result.errors:
        print(f"  ERROR: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify processed preprocessing outputs.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        action="append",
        help="Dataset to verify. Repeat for multiple datasets. Defaults to all datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset or list(DATASETS)
    results = [verify_dataset(dataset) for dataset in datasets]
    for result in results:
        print_result(result)

    n_errors = sum(len(result.errors) for result in results)
    n_warnings = sum(len(result.warnings) for result in results)
    print(f"\nVerification complete: {n_errors} errors, {n_warnings} warnings")
    if n_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
