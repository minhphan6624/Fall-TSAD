from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.model_performance_common import OUT_DIR, load_tracker, model_label


COLORS = {"classification": "#4C78A8", "tsad": "#F58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one metric with one subplot per dataset."
    )
    parser.add_argument("--metric", default="auprc_mean")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", choices=("classification", "tsad"), default=None)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def filtered_tracker(args: argparse.Namespace):
    df = load_tracker()
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
    if args.modes:
        df = df[df["mode"].isin(args.modes)]
    if args.models:
        df = df[df["model"].isin(args.models)]
    if df.empty:
        raise ValueError("No tracker rows match the requested filters.")
    if args.metric not in df.columns:
        raise ValueError(f"Tracker does not contain metric column: {args.metric}")
    return df.copy()


def next_output_path(args: argparse.Namespace):
    dataset_part = "_".join(args.datasets) if args.datasets else "all_datasets"
    mode_part = "_".join(args.modes) if args.modes else "all_modes"
    model_part = "_".join(args.models) if args.models else "all_models"
    stem = f"dataset_subplots_{dataset_part}_{mode_part}_{model_part}_{args.metric}"
    out = OUT_DIR / f"{stem}.png"

    counter = 2
    while out.exists():
        out = OUT_DIR / f"{stem}_{counter}.png"
        counter += 1
    return out


def plot_metric_dataset_subplots(args: argparse.Namespace):
    df = filtered_tracker(args)
    datasets = list(dict.fromkeys(df["dataset"]))
    n_cols = 2 if len(datasets) > 1 else 1
    n_rows = math.ceil(len(datasets) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.6 * n_rows),
        squeeze=False,
        sharey=True,
    )

    y_max = min(1.05, max(1.0, float(df[args.metric].max()) * 1.15))
    std_col = args.metric.replace("_mean", "_std")

    for ax, dataset in zip(axes.ravel(), datasets):
        rows = df[df["dataset"] == dataset]
        x = np.arange(len(rows))
        labels = [model_label(model) for model in rows["model"]]
        colors = rows["mode"].map(COLORS).to_numpy()
        yerr = rows[std_col].to_numpy() if std_col in rows.columns else None

        ax.bar(x, rows[args.metric].to_numpy(), yerr=yerr, capsize=3, color=colors)
        ax.set_title(dataset)
        ax.set_ylabel(args.metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["classification"], label="classification"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["tsad"], label="tsad"),
    ]
    fig.suptitle(args.metric.replace("_", " ").upper(), y=0.99)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out = next_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_metric_dataset_subplots(parse_args())
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
