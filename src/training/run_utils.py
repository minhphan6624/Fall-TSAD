import json
from pathlib import Path

import pandas as pd

from src.training.extract_features import extract_features


def features_for_split(split):
    rates = split["meta"]["sampling_rate_hz"].to_numpy()
    return extract_features(split["X"], sampling_rate_hz=rates)


def make_run_dir(run_root, dataset, mode, model_name, model_seed):
    return Path(run_root) / dataset / mode / model_name / f"model_seed_{model_seed}"


def save_json(path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_predictions(path, meta, y_true, scores, threshold):
    predictions = meta[
        ["window_id", "subject_id", "activity_id", "trial_id", "split", "window_label"]
    ].copy()
    predictions["y_true"] = y_true
    predictions["score"] = scores
    predictions["pred_label"] = (scores >= threshold).astype(int)
    predictions.to_csv(path, index=False)


def save_feature_importance(path, feature_names, importances):
    rows = pd.DataFrame({"feature": feature_names, "importance": importances})
    rows = rows.sort_values("importance", ascending=False)
    rows.to_csv(path, index=False)
