import argparse
import json
import pickle

from sklearn.ensemble import RandomForestClassifier

from src.training.data import label_counts, load_window_data
from src.training.evaluation import compute_binary_metrics
from src.training.run_utils import (
    features_for_split,
    make_run_dir,
    save_feature_importance,
    save_json,
    save_predictions,
)
from src.training.thresholds import find_best_f1_threshold

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Random Forest baseline.")
    parser.add_argument("--dataset", default="sisfall")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--run-root", default="runs/benchmark")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()

    data = load_window_data(args.dataset, "classification", data_root=args.data_root)

    X_train, feature_names = features_for_split(data["train"])
    X_val, _ = features_for_split(data["val"])
    X_test, _ = features_for_split(data["test"])

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.model_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    threshold = find_best_f1_threshold(y_val, val_scores)
    val_metrics = compute_binary_metrics(y_val, val_scores, threshold)
    test_metrics = compute_binary_metrics(y_test, test_scores, threshold)

    run_dir = make_run_dir(
        args.run_root, args.dataset, "classification", "random_forest", args.model_seed
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", vars(args))
    save_json(run_dir / "feature_names.json", feature_names)
    save_json(
        run_dir / "metrics.json",
        {
            "train_label_counts": label_counts(y_train),
            "val": val_metrics,
            "test": test_metrics,
        },
    )

    save_predictions(run_dir / "predictions_val.csv", data["val"]["meta"], y_val, val_scores, threshold)
    save_predictions(run_dir / "predictions_test.csv", data["test"]["meta"], y_test, test_scores, threshold)
    save_feature_importance(run_dir / "feature_importance.csv", feature_names, model.feature_importances_)

    # Save model
    with (run_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved Random Forest run to {run_dir}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
