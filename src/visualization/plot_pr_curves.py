from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.visualization.model_performance_common import (
    BEST_MAIN_MODELS,
    MAIN_BENCHMARK_PREFIX,
    OUT_DIR,
    load_cv_predictions,
    model_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark precision-recall curves from predictions_test.csv files."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name in the tracker, e.g. sisfall.")
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="Classifier model to plot. Defaults to the hard-coded best main-benchmark classifier.",
    )
    parser.add_argument(
        "--tsad-model",
        default=None,
        help="TSAD model to plot. Defaults to the hard-coded best main-benchmark TSAD model.",
    )
    return parser.parse_args()


def plot_pr_curves(args: argparse.Namespace):
    if args.dataset not in BEST_MAIN_MODELS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Expected one of {sorted(BEST_MAIN_MODELS)}.")

    best_models = BEST_MAIN_MODELS[args.dataset]
    models = [
        ("classification", args.classifier_model or best_models["classification"]),
        ("tsad", args.tsad_model or best_models["tsad"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 5.2))
    first_df = None
    for mode, model in models:
        df = load_cv_predictions(args.dataset, mode, model)
        if first_df is None:
            first_df = df

        precision, recall, _ = precision_recall_curve(df["y_true"], df["score"])
        ap = average_precision_score(df["y_true"], df["score"])
        ax.plot(
            recall,
            precision,
            linewidth=2.0,
            label=f"{model_label(model)} ({mode}, AP={ap:.3f})",
        )

    positive_rate = float(first_df["y_true"].mean())
    ax.axhline(
        positive_rate,
        color="0.55",
        linestyle="--",
        linewidth=1.0,
        label=f"Prevalence={positive_rate:.3f}",
    )
    ax.set_title(f"{args.dataset} precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    dataset_prefix = MAIN_BENCHMARK_PREFIX.format(dataset=args.dataset)
    out = OUT_DIR / f"{dataset_prefix}_pr_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_pr_curves(parse_args())
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
