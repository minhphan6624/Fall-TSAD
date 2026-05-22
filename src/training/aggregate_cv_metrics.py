import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.run_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate test metrics across CV folds.")
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--mode", required=True, choices=("classification", "tsad"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--run-root", default="runs/benchmark")
    return parser.parse_args()


def metrics_path(run_root: Path, dataset: str, mode: str, model: str, model_seed: int) -> Path:
    return run_root / dataset / mode / model / f"model_seed_{model_seed}" / "metrics.json"


def load_fold_test_metrics(
    run_root: Path,
    dataset_prefix: str,
    n_folds: int,
    mode: str,
    model: str,
    model_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing_paths: list[Path] = []

    for fold_index in range(n_folds):
        dataset = f"{dataset_prefix}_fold{fold_index}"
        path = metrics_path(run_root, dataset, mode, model, model_seed)
        if not path.exists():
            missing_paths.append(path)
            continue

        with path.open() as f:
            metrics = json.load(f)

        if "test" not in metrics:
            raise ValueError(f"Missing 'test' section in {path}")

        row = {"fold_index": fold_index, "dataset": dataset}
        row.update(metrics["test"])
        rows.append(row)

    if missing_paths:
        formatted = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing fold metric files:\n{formatted}")

    return rows


def summarize_numeric_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"fold_index", "dataset"} and isinstance(value, (int, float))
        }
    )

    summary_rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if isinstance(row.get(metric_name), (int, float)) and row.get(metric_name) is not None
        ]
        if not values:
            continue

        arr = np.asarray(values, dtype=np.float64)
        summary_rows.append(
            {
                "metric": metric_name,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n_folds": int(len(arr)),
            }
        )

    return summary_rows


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)

    fold_metrics = load_fold_test_metrics(
        run_root=run_root,
        dataset_prefix=args.dataset_prefix,
        n_folds=args.n_folds,
        mode=args.mode,
        model=args.model,
        model_seed=args.model_seed,
    )
    summary_rows = summarize_numeric_metrics(fold_metrics)

    out_dir = (
        run_root
        / f"{args.dataset_prefix}_cv"
        / args.mode
        / args.model
        / f"model_seed_{args.model_seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        out_dir / "cv_metrics.json",
        {
            "dataset_prefix": args.dataset_prefix,
            "n_folds": args.n_folds,
            "mode": args.mode,
            "model": args.model,
            "model_seed": args.model_seed,
            "summary": summary_rows,
            "fold_metrics": fold_metrics,
        },
    )
    pd.DataFrame(summary_rows).to_csv(out_dir / "cv_metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(out_dir / "fold_metrics.csv", index=False)

    print(f"Saved CV metrics to {out_dir}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
