import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.training.data import label_counts, load_window_data
from src.training.evaluation import compute_binary_metrics
from src.training.thresholds import find_best_f1_threshold
from src.training.extract_features import extract_features

def _features_for_split(split):
    rates = split["meta"]["sampling_rate_hz"].to_numpy()
    return extract_features(split["X"], sampling_rate_hz=rates)

def _save_json(path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _save_predictions(path, meta, y_true, scores, threshold):
    predictions = meta[
        ["window_id", "subject_id", "activity_id", "trial_id", "split", "window_label"]
    ].copy()
    predictions["y_true"] = y_true
    predictions["score"] = scores
    predictions["pred_label"] = (scores >= threshold).astype(int)
    predictions.to_csv(path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Isolation Forest TSAD baseline.")
    parser.add_argument("--dataset", default="sisfall")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--run-root", default="runs/benchmark")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--max-features", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    data = load_window_data(args.dataset, "tsad", data_root=args.data_root)

    X_train, feature_names = _features_for_split(data["train"])
    X_val, _ = _features_for_split(data["val"])
    X_test, _ = _features_for_split(data["test"])

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination if args.contamination == "auto" else float(args.contamination),
        max_samples=args.max_samples if args.max_samples == "auto" else float(args.max_samples),
        max_features=args.max_features,
        random_state=args.model_seed,
        n_jobs=-1,
    )
    model.fit(X_train)

    val_scores = -model.decision_function(X_val)
    test_scores = -model.decision_function(X_test)

    threshold = find_best_f1_threshold(y_val, val_scores)
    val_metrics = compute_binary_metrics(y_val, val_scores, threshold)
    test_metrics = compute_binary_metrics(y_test, test_scores, threshold)

    run_dir = Path(args.run_root) / args.dataset / "tsad" / "isolation_forest" / f"model_seed_{args.model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_json(run_dir / "config.json", vars(args))
    _save_json(run_dir / "feature_names.json", feature_names)
    _save_json(
        run_dir / "metrics.json",
        {
            "train_label_counts": label_counts(y_train),
            "val": val_metrics,
            "test": test_metrics,
        },
    )
    _save_predictions(run_dir / "predictions_val.csv", data["val"]["meta"], y_val, val_scores, threshold)
    _save_predictions(run_dir / "predictions_test.csv", data["test"]["meta"], y_test, test_scores, threshold)

    with (run_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved Isolation Forest run to {run_dir}")
    print(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()
