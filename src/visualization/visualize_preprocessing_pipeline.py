from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.preprocessing.label_windows import (
    FALL_LABEL_THRESHOLD,
    compute_trial_label_info,
    overlap_ratio,
)
from src.preprocessing.window_trials import (
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SECONDS,
    compute_window_geometry,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUT_DIR = ROOT_DIR / "figures" / "preprocessing"
PICKLE_NAMES = {
    "sisfall": "SisFall.pkl",
    "umafall": "UMAFall.pkl",
    "fallalld": "FallAllD.pkl",
    "upfall": "UP-FALL.pkl",
}
SMV_COLOR = "black"
IMPACT_COLOR = "tab:red"
FALL_REGION_COLOR = "tab:orange"
FALL_WINDOW_COLOR = "tab:red"
NON_FALL_WINDOW_COLOR = "tab:blue"


def plot_legend_handles(include_windows: bool = False) -> list[object]:
    handles: list[object] = [
        Line2D([0], [0], color=SMV_COLOR, linewidth=1.0, label="SMV"),
        patches.Patch(facecolor=FALL_REGION_COLOR, alpha=0.18, label="fall region"),
        Line2D([0], [0], color=IMPACT_COLOR, linewidth=1.2, label="impact"),
    ]
    if include_windows:
        handles.extend(
            [
                patches.Patch(
                    facecolor=NON_FALL_WINDOW_COLOR,
                    edgecolor=NON_FALL_WINDOW_COLOR,
                    alpha=0.35,
                    label="non-fall window",
                ),
                patches.Patch(
                    facecolor=FALL_WINDOW_COLOR,
                    edgecolor=FALL_WINDOW_COLOR,
                    alpha=0.35,
                    label="fall window",
                ),
            ]
        )
    return handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize window segmentation and labeling for one trial.")
    parser.add_argument("--dataset", default="sisfall", choices=sorted(PICKLE_NAMES))
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--activity-id", default=None)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--zoom-padding-seconds", type=float, default=3.0)
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_trial(args: argparse.Namespace) -> pd.Series:
    path = args.interim_dir / args.dataset / PICKLE_NAMES[args.dataset]
    df = pd.read_pickle(path)

    mask = df["is_fall"].astype(int) == 1
    for column, value in (
        ("subject_id", args.subject_id),
        ("activity_id", args.activity_id),
        ("trial_id", args.trial_id),
    ):
        if value is not None:
            mask &= df[column].astype(str) == str(value)

    matches = df[mask]
    if matches.empty:
        raise ValueError("No matching fall trial found.")

    return matches.iloc[0]


def make_windows(trial: pd.Series, window_seconds: float, overlap: float) -> pd.DataFrame:
    fs = float(trial["sampling_rate_hz"])
    n_samples = int(trial["n_samples"])
    window_len, stride = compute_window_geometry(fs, window_seconds, overlap)
    label_info = compute_trial_label_info(trial)

    rows = []
    for start_idx in range(0, n_samples - window_len + 1, stride):
        end_idx = start_idx + window_len
        ratio = overlap_ratio(start_idx, end_idx, label_info["fall_start"], label_info["fall_end"])
        rows.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_sec": start_idx / fs,
                "end_sec": end_idx / fs,
                "label": int(ratio > FALL_LABEL_THRESHOLD),
                "overlap_ratio": ratio,
            }
        )

    return pd.DataFrame(rows)


def add_window_patches(ax: plt.Axes, windows: pd.DataFrame, ymin: float, ymax: float) -> None:
    height = ymax - ymin
    for _, window in windows.iterrows():
        color = FALL_WINDOW_COLOR if int(window["label"]) == 1 else NON_FALL_WINDOW_COLOR
        rect = patches.Rectangle(
            (window["start_sec"], ymin),
            window["end_sec"] - window["start_sec"],
            height,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=1.0,
        )
        ax.add_patch(rect)


def trial_plot_data(trial: pd.Series, args: argparse.Namespace) -> dict[str, object]:
    acc = np.asarray(trial["acc"], dtype=np.float32)
    fs = float(trial["sampling_rate_hz"])
    smv = np.linalg.norm(acc, axis=1)
    t = np.arange(acc.shape[0]) / fs

    label_info = compute_trial_label_info(trial)
    impact_sec = label_info["impact_idx"] / fs
    fall_start_sec = label_info["fall_start"] / fs
    fall_end_sec = label_info["fall_end"] / fs
    zoom_start = max(0.0, fall_start_sec - args.zoom_padding_seconds)
    zoom_end = min(t[-1], fall_end_sec + args.zoom_padding_seconds)

    return {
        "smv": smv,
        "time": t,
        "impact_sec": impact_sec,
        "fall_start_sec": fall_start_sec,
        "fall_end_sec": fall_end_sec,
        "zoom_start": zoom_start,
        "zoom_end": zoom_end,
    }


def output_stem(trial: pd.Series, dataset: str) -> str:
    return f"{dataset}_{trial['subject_id']}_{trial['activity_id']}_{trial['trial_id']}"


def plot_trial_overview(trial: pd.Series, args: argparse.Namespace) -> Path:
    data = trial_plot_data(trial, args)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(data["time"], data["smv"], color=SMV_COLOR, linewidth=1.0)
    ax.axvspan(
        data["fall_start_sec"],
        data["fall_end_sec"],
        color=FALL_REGION_COLOR,
        alpha=0.18,
        label="fall region",
    )
    ax.axvline(data["impact_sec"], color=IMPACT_COLOR, linewidth=1.2, label="impact")
    ax.set_title(
        f"{args.dataset}: subject={trial['subject_id']} activity={trial['activity_id']} trial={trial['trial_id']}"
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SMV")
    ax.grid(alpha=0.25)
    ax.legend(handles=plot_legend_handles(), loc="upper right")

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{output_stem(trial, args.dataset)}_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_window_labeling_detail(trial: pd.Series, windows: pd.DataFrame, args: argparse.Namespace) -> Path:
    data = trial_plot_data(trial, args)
    zoom_start = float(data["zoom_start"])
    zoom_end = float(data["zoom_end"])
    zoom_windows = windows[(windows["end_sec"] >= zoom_start) & (windows["start_sec"] <= zoom_end)]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 5.4),
        sharex=False,
        gridspec_kw={"height_ratios": [2.8, 0.8]},
    )
    zoom_ax, label_ax = axes

    zoom_ax.plot(data["time"], data["smv"], color=SMV_COLOR, linewidth=1.0)
    zoom_ax.axvspan(data["fall_start_sec"], data["fall_end_sec"], color=FALL_REGION_COLOR, alpha=0.18)
    zoom_ax.axvline(data["impact_sec"], color=IMPACT_COLOR, linewidth=1.2)
    y_min, y_max = zoom_ax.get_ylim()
    add_window_patches(zoom_ax, zoom_windows, y_min, y_max)
    zoom_ax.set_xlim(zoom_start, zoom_end)
    zoom_ax.set_title("Window segmentation around fall region")
    zoom_ax.set_ylabel("SMV")
    zoom_ax.grid(alpha=0.25)
    zoom_ax.legend(handles=plot_legend_handles(include_windows=True), loc="upper right", ncol=3)

    for i, (_, window) in enumerate(zoom_windows.iterrows()):
        color = FALL_WINDOW_COLOR if int(window["label"]) == 1 else NON_FALL_WINDOW_COLOR
        label_ax.broken_barh(
            [(window["start_sec"], window["end_sec"] - window["start_sec"])],
            (i % 2, 0.8),
            facecolors=color,
            edgecolors=color,
            alpha=0.7,
        )

    label_ax.axvspan(data["fall_start_sec"], data["fall_end_sec"], color=FALL_REGION_COLOR, alpha=0.14)
    label_ax.set_xlim(zoom_start, zoom_end)
    label_ax.set_ylim(-0.2, 2.2)
    label_ax.set_yticks([0.4, 1.4])
    label_ax.set_yticklabels(["staggered\nwindows", "staggered\nwindows"])
    label_ax.set_xlabel("time (s)")
    label_ax.set_title(f"window label: fall if overlap with fall region > {FALL_LABEL_THRESHOLD:.2f}")
    label_ax.legend(
        handles=[
            patches.Patch(
                facecolor=NON_FALL_WINDOW_COLOR,
                edgecolor=NON_FALL_WINDOW_COLOR,
                alpha=0.7,
                label="label 0: non-fall",
            ),
            patches.Patch(
                facecolor=FALL_WINDOW_COLOR,
                edgecolor=FALL_WINDOW_COLOR,
                alpha=0.7,
                label="label 1: fall",
            ),
            patches.Patch(facecolor=FALL_REGION_COLOR, alpha=0.14, label="fall region"),
        ],
        loc="upper right",
        ncol=3,
    )

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{output_stem(trial, args.dataset)}_window_labeling.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    trial = load_trial(args)
    windows = make_windows(trial, args.window_seconds, args.overlap)
    for out_path in (
        plot_trial_overview(trial, args),
        plot_window_labeling_detail(trial, windows, args),
    ):
        print(out_path)


if __name__ == "__main__":
    main()
