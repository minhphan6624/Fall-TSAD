from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.model_performance_common import EASY_ADL_REPORT_PATH, OUT_DIR


COLORS = {"classification": "#4C78A8", "tsad": "#F58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot easy-ADL delta AUPRC.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def slug(values: list[str] | None, default: str) -> str:
    if not values:
        return default
    return "_".join(value.lower().replace(" ", "_").replace("-", "") for value in values)


def next_output_path(args: argparse.Namespace):
    stem = f"easy_adl_delta_auprc_{slug(args.datasets, 'all_datasets')}_{slug(args.models, 'all_models')}"
    out = OUT_DIR / f"{stem}.png"

    counter = 2
    while out.exists():
        out = OUT_DIR / f"{stem}_{counter}.png"
        counter += 1
    return out


def plot_easy_adl_delta_auprc(args: argparse.Namespace):
    df = pd.read_csv(EASY_ADL_REPORT_PATH)
    if args.datasets:
        df = df[df["Dataset"].str.lower().isin(args.datasets)]
    if args.models:
        df = df[df["Model"].isin(args.models)]
    if df.empty:
        raise ValueError("No easy-ADL rows match the requested filters.")

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

    max_abs_delta = max(0.05, float(df["Delta AUPRC"].abs().max()) * 1.15)
    for ax, dataset in zip(axes.ravel(), datasets):
        rows = df[df["Dataset"] == dataset]
        x = np.arange(len(rows))
        colors = rows["Mode"].map(COLORS).to_numpy()

        ax.bar(x, rows["Delta AUPRC"].to_numpy(), color=colors)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(dataset)
        ax.set_ylabel("Delta AUPRC")
        ax.set_xticks(x)
        ax.set_xticklabels(rows["Model"], rotation=35, ha="right")
        ax.set_ylim(-max_abs_delta, max_abs_delta)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["classification"], label="classification"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["tsad"], label="tsad"),
    ]
    fig.suptitle("Easy-ADL ablation: Delta AUPRC", y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out = next_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_easy_adl_delta_auprc(parse_args())
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
