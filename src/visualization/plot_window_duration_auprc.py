from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.model_performance_common import (
    OUT_DIR,
    REPRESENTATIVE_WINDOW_MODELS,
    WINDOW_DURATION_REPORT_PATH,
)


WINDOWS = ("2s", "3s", "5s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AUPRC across tested window sizes for representative models."
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=list(REPRESENTATIVE_WINDOW_MODELS))
    return parser.parse_args()


def slug(values: list[str] | None, default: str) -> str:
    if not values:
        return default
    return "_".join(value.lower().replace(" ", "_").replace("-", "") for value in values)


def next_output_path(args: argparse.Namespace):
    dataset_part = slug(args.datasets, "all_datasets")
    model_part = slug(args.models, "all_models")
    stem = f"window_duration_auprc_{dataset_part}_{model_part}"
    out = OUT_DIR / f"{stem}.png"

    counter = 2
    while out.exists():
        out = OUT_DIR / f"{stem}_{counter}.png"
        counter += 1
    return out


def plot_window_duration_auprc(args: argparse.Namespace):
    df = pd.read_csv(WINDOW_DURATION_REPORT_PATH)
    if args.datasets:
        df = df[df["Dataset"].str.lower().isin(args.datasets)]
    if args.models:
        df = df[df["Model"].isin(args.models)]
    if df.empty:
        raise ValueError("No report rows match the requested filters.")

    datasets = list(dict.fromkeys(df["Dataset"]))
    n_cols = 2 if len(datasets) > 1 else 1
    n_rows = math.ceil(len(datasets) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.5 * n_rows),
        squeeze=False,
        sharey=True,
    )

    y_max = min(1.05, max(1.0, float(df[[f"{window} AUPRC" for window in WINDOWS]].max().max()) * 1.08))
    for ax, dataset in zip(axes.ravel(), datasets):
        rows = df[df["Dataset"] == dataset]
        for _, row in rows.iterrows():
            values = [row[f"{window} AUPRC"] for window in WINDOWS]
            ax.plot(WINDOWS, values, marker="o", linewidth=2.0, label=row["Model"])

        ax.set_title(dataset)
        ax.set_xlabel("Window size")
        ax.set_ylabel("AUPRC")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")

    fig.suptitle("AUPRC across window duration", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = next_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_window_duration_auprc(parse_args())
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
