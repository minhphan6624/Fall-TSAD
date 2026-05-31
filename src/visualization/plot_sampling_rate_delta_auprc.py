from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.model_performance_common import OUT_DIR, SAMPLING_RATE_REPORT_PATH


DATASETS = ("sisfall", "fallalld")
COLORS = {"classification": "#4C78A8", "tsad": "#F58518"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sampling-rate delta AUPRC for SisFall and FallAllD."
    )
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def next_output_path(args: argparse.Namespace):
    model_part = (
        "_".join(model.lower().replace(" ", "_").replace("-", "") for model in args.models)
        if args.models
        else "all_models"
    )
    stem = f"sampling_rate_delta_auprc_sisfall_fallalld_{model_part}"
    out = OUT_DIR / f"{stem}.png"

    counter = 2
    while out.exists():
        out = OUT_DIR / f"{stem}_{counter}.png"
        counter += 1
    return out


def plot_sampling_rate_delta_auprc(args: argparse.Namespace):
    df = pd.read_csv(SAMPLING_RATE_REPORT_PATH)
    df = df[df["Dataset"].str.lower().isin(DATASETS)].copy()
    if args.models:
        df = df[df["Model"].isin(args.models)]
    if df.empty:
        raise ValueError("No sampling-rate rows match the requested filters.")

    datasets = ["SisFall", "FallAllD"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    max_abs_delta = max(0.05, float(df["Delta AUPRC"].abs().max()) * 1.15)

    for ax, dataset in zip(axes, datasets):
        rows = df[df["Dataset"] == dataset].iloc[::-1]
        y = np.arange(len(rows))
        colors = rows["Mode"].map(COLORS).to_numpy()

        ax.barh(y, rows["Delta AUPRC"].to_numpy(), color=colors)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_title(dataset)
        ax.set_xlabel("Delta AUPRC (20 Hz - native)")
        ax.set_yticks(y)
        ax.set_yticklabels(rows["Model"])
        ax.set_xlim(-max_abs_delta, max_abs_delta)
        ax.grid(axis="x", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["classification"], label="classification"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["tsad"], label="tsad"),
    ]
    fig.suptitle("Sampling-rate ablation: Delta AUPRC", y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out = next_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_sampling_rate_delta_auprc(parse_args())
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
