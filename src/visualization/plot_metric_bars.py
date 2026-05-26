from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.model_performance_common import (
    OUT_DIR,
    load_tracker,
    model_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark metric comparison bars from a results tracker CSV."
    )
    parser.add_argument("--metrics", nargs="+", default=["auprc_mean"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", choices=("classification", "tsad"), default=None)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def filter_tracker(args: argparse.Namespace) -> pd.DataFrame:
    df = load_tracker()
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
    if args.modes:
        df = df[df["mode"].isin(args.modes)]
    if args.models:
        df = df[df["model"].isin(args.models)]
    if df.empty:
        raise ValueError("No tracker rows match the requested filters.")
    return df.copy()


def filename_part(values: list[str] | None, default: str) -> str:
    return "_".join(values) if values else default


def next_output_path(args: argparse.Namespace):
    metric_name = "_".join(args.metrics)
    dataset_name = filename_part(args.datasets, "all_datasets")
    mode_name = filename_part(args.modes, "all_modes")
    model_name = filename_part(args.models, "all_models")
    stem = f"{dataset_name}_{mode_name}_{model_name}_{metric_name}"
    out = OUT_DIR / f"{stem}.png"

    counter = 2
    while out.exists():
        out = OUT_DIR / f"{stem}_{counter}.png"
        counter += 1
    return out


def plot_metric_bars(args: argparse.Namespace):
    df = filter_tracker(args)
    missing_metrics = [metric for metric in args.metrics if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"Tracker does not contain metric columns: {missing_metrics}")

    fig, axes = plt.subplots(
        1,
        len(args.metrics),
        figsize=(max(7, 4.8 * len(args.metrics)), 5.2),
        squeeze=False,
    )

    for ax, metric in zip(axes[0], args.metrics):
        labels = [
            f"{row.dataset}\n{model_label(row.model)}"
            if df["dataset"].nunique() > 1
            else model_label(row.model)
            for row in df.itertuples()
        ]
        x = np.arange(len(df))
        colors = df["mode"].map({"classification": "#4C78A8", "tsad": "#F58518"}).to_numpy()
        std_col = metric.replace("_mean", "_std")
        yerr = df[std_col].to_numpy() if std_col in df.columns else None

        ax.bar(x, df[metric].to_numpy(), yerr=yerr, capsize=3, color=colors)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, min(1.05, max(1.0, float(df[metric].max()) * 1.15)))
        ax.grid(axis="y", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="classification"),
        plt.Rectangle((0, 0), 1, 1, color="#F58518", label="tsad"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = next_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_metric_bars(parse_args())
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
