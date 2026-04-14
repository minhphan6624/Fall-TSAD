from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
DEFAULT_OUT_DIR = ROOT_DIR / "figures" / "interim_eda"
DEFAULT_DATASETS = ("sisfall", "umafall", "fallalld", "upfall")
PICKLE_NAMES = {
    "sisfall": "SisFall.pkl",
    "umafall": "UMAFall.pkl",
    "fallalld": "FallAllD.pkl",
    "upfall": "UP-FALL.pkl",
}
DISPLAY_NAMES = {
    "sisfall": "SisFall",
    "umafall": "UMAFall",
    "fallalld": "FallAllD",
    "upfall": "UP-FALL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EDA plots from the interim trial pickles."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(PICKLE_NAMES.keys()),
        default=list(DEFAULT_DATASETS),
        help="Subset of datasets to visualize.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_DIR,
        help="Directory containing dataset subfolders with interim pickles.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where plots and summary CSV are written.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=3,
        help="Number of representative fall and non-fall trials per dataset.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=8.0,
        help="Maximum number of seconds to show for each example plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when sampling example trials.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help=(
            "Plot per-trial z-scored axes instead of raw amplitudes. "
            "Useful when comparing shapes across datasets with different units."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=("start", "peak"),
        default="start",
        help=(
            "How to choose the plotted snippet. "
            "'start' shows the first N seconds, 'peak' centers the snippet on the most active region."
        ),
    )
    return parser.parse_args()


def interim_pickle_path(interim_dir: Path, dataset: str) -> Path:
    return interim_dir / dataset / PICKLE_NAMES[dataset]


def load_dataset(interim_dir: Path, dataset: str) -> pd.DataFrame:
    path = interim_pickle_path(interim_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Could not find interim pickle: {path}")

    df = pd.read_pickle(path).copy()
    df["duration_sec"] = df["n_samples"] / df["sampling_rate_hz"]
    df["label_name"] = df["is_fall"].map({0: "non_fall", 1: "fall"})
    return df


def summarize_dataset(df: pd.DataFrame, dataset: str) -> dict[str, object]:
    durations = df["duration_sec"]
    label_counts = df["label_name"].value_counts()

    return {
        "dataset": dataset,
        "display_name": DISPLAY_NAMES[dataset],
        "n_trials": int(len(df)),
        "n_fall": int(label_counts.get("fall", 0)),
        "n_non_fall": int(label_counts.get("non_fall", 0)),
        "sampling_rate_hz": float(df["sampling_rate_hz"].iloc[0]),
        "duration_min_sec": float(durations.min()),
        "duration_median_sec": float(durations.median()),
        "duration_max_sec": float(durations.max()),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_activities": int(df["activity_id"].nunique()),
        "amplitude_note": amplitude_note(dataset),
    }


def amplitude_note(dataset: str) -> str:
    if dataset in {"sisfall", "upfall"}:
        return "raw amplitudes are in g"
    if dataset == "fallalld":
        return "raw amplitudes are preserved; do not compare absolute scale across datasets"
    return "parser stores waist accelerometer values without a dataset-wide conversion step"


def select_examples(
    df: pd.DataFrame,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for is_fall in (0, 1):
        group = df[df["is_fall"] == is_fall]
        if group.empty:
            continue

        n_select = min(samples_per_class, len(group))
        sampled = group.sample(n=n_select, random_state=seed)
        groups.append(sampled.sort_values(["subject_id", "activity_id", "trial_id"]))

    if not groups:
        return df.iloc[0:0].copy()

    return pd.concat(groups, axis=0, ignore_index=True)


def maybe_standardize(acc: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return acc

    mean = acc.mean(axis=0, keepdims=True)
    std = acc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (acc - mean) / std


def make_time_axis(n_samples: int, fs_hz: float, max_seconds: float) -> np.ndarray:
    n_keep = min(n_samples, max(1, int(round(max_seconds * fs_hz))))
    return np.arange(n_keep) / fs_hz


def select_window(
    acc: np.ndarray,
    fs_hz: float,
    max_seconds: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_samples = acc.shape[0]
    n_keep = min(n_samples, max(1, int(round(max_seconds * fs_hz))))

    if mode == "start" or n_keep >= n_samples:
        start_idx = 0
    else:
        motion = np.linalg.norm(np.diff(acc, axis=0, prepend=acc[[0]]), axis=1)
        kernel = np.ones(n_keep, dtype=np.float32) / float(n_keep)
        scores = np.convolve(motion, kernel, mode="same")
        peak_idx = int(np.argmax(scores))
        start_idx = max(0, min(n_samples - n_keep, peak_idx - (n_keep // 2)))

    end_idx = start_idx + n_keep
    time_axis = np.arange(start_idx, end_idx) / fs_hz
    return acc[start_idx:end_idx], time_axis, start_idx / fs_hz, end_idx / fs_hz


def plot_examples(
    df: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    samples_per_class: int,
    max_seconds: float,
    seed: int,
    standardize: bool,
    window_mode: str,
) -> Path:
    examples = select_examples(df, samples_per_class=samples_per_class, seed=seed)
    if examples.empty:
        raise ValueError(f"No examples available for dataset: {dataset}")

    n_rows = len(examples)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(13, max(3.2 * n_rows, 4.5)),
        sharex=False,
        squeeze=False,
    )

    ylabel = "z-score" if standardize else "acc"
    note = "standardized per trial" if standardize else amplitude_note(dataset)
    fig.suptitle(
        f"{DISPLAY_NAMES[dataset]} waist accelerometer examples\n{note} | window_mode={window_mode}",
        fontsize=14,
        y=0.995,
    )

    for ax, (_, row) in zip(axes[:, 0], examples.iterrows(), strict=False):
        acc = maybe_standardize(row["acc"], enabled=standardize)
        window, time_axis, start_sec, end_sec = select_window(
            acc=acc,
            fs_hz=float(row["sampling_rate_hz"]),
            max_seconds=max_seconds,
            mode=window_mode,
        )
        smv = np.linalg.norm(window, axis=1)

        ax.plot(time_axis, window[:, 0], linewidth=1.0, label="x")
        ax.plot(time_axis, window[:, 1], linewidth=1.0, label="y")
        ax.plot(time_axis, window[:, 2], linewidth=1.0, label="z")
        ax.plot(time_axis, smv, linewidth=1.2, linestyle="--", color="black", label="smv")

        label_name = "fall" if int(row["is_fall"]) == 1 else "non-fall"
        ax.set_title(
            (
                f"{label_name} | subject={row['subject_id']} | activity={row['activity_id']} "
                f"| trial={row['trial_id']} | fs={row['sampling_rate_hz']:.1f} Hz "
                f"| window={start_sec:.1f}-{end_sec:.1f}s"
            ),
            fontsize=10,
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=4, fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_standardized" if standardize else ""
    suffix = f"{suffix}_{window_mode}"
    out_path = out_dir / f"{dataset}_examples{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_duration_histogram(df: pd.DataFrame, dataset: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for is_fall, color, label in (
        (0, "tab:blue", "non-fall"),
        (1, "tab:red", "fall"),
    ):
        subset = df[df["is_fall"] == is_fall]["duration_sec"]
        if subset.empty:
            continue
        ax.hist(subset, bins=30, alpha=0.55, label=label, color=color)

    ax.set_title(f"{DISPLAY_NAMES[dataset]} trial duration distribution")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_duration_hist.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(rows: list[dict[str, object]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "interim_summary.csv"
    pd.DataFrame(rows).sort_values("dataset").to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    summary_rows: list[dict[str, object]] = []

    print("Creating interim EDA outputs...")
    for dataset in args.datasets:
        df = load_dataset(args.interim_dir, dataset)
        summary_rows.append(summarize_dataset(df, dataset))

        examples_path = plot_examples(
            df=df,
            dataset=dataset,
            out_dir=out_dir,
            samples_per_class=args.samples_per_class,
            max_seconds=args.max_seconds,
            seed=args.seed,
            standardize=args.standardize,
            window_mode=args.window_mode,
        )
        duration_path = plot_duration_histogram(df=df, dataset=dataset, out_dir=out_dir)

        print(f"- {dataset}: {examples_path}")
        print(f"- {dataset}: {duration_path}")

    summary_path = write_summary(summary_rows, out_dir=out_dir)
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
