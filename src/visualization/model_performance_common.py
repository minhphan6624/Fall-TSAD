from pathlib import Path
import pandas as pd


RUN_ROOT = Path("runs/benchmark")
TRACKER_PATH = Path("docs/results_trackers/main_benchmark.csv")
OUT_DIR = Path("figures/model_performance")
MODEL_SEED = 42
N_FOLDS = 5
MAIN_BENCHMARK_PREFIX = "{dataset}_20hz_2s"
BEST_MAIN_MODELS = {
    "fallalld": {"classification": "xgboost", "tsad": "dense_ae"},
    "sisfall": {"classification": "xgboost", "tsad": "isolation_forest"},
    "umafall": {"classification": "xgboost", "tsad": "dense_ae"},
    "upfall": {"classification": "xgboost", "tsad": "isolation_forest"},
}
DISPLAY_NAMES = {
    "cnn1d": "CNN1D",
    "cnn1d_ae": "CNN1D AE",
    "cnn1d_ae_large": "CNN1D AE Large",
    "cnn1d_large": "CNN1D Large",
    "dense_ae": "Dense AE",
    "isolation_forest": "Isolation Forest",
    "lstm_ae": "LSTM AE",
    "lstm_classifier": "LSTM Classifier",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def model_label(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.replace("_", " ").title())


def load_tracker() -> pd.DataFrame:
    if not TRACKER_PATH.exists():
        raise FileNotFoundError(f"Could not find tracker CSV: {TRACKER_PATH}")
    return pd.read_csv(TRACKER_PATH)


def load_cv_predictions(dataset: str, mode: str, model: str) -> pd.DataFrame:
    dataset_prefix = MAIN_BENCHMARK_PREFIX.format(dataset=dataset)
    rows = []
    missing = []
    for fold in range(N_FOLDS):
        path = ( RUN_ROOT / f"{dataset_prefix}_fold{fold}" / mode / model
            / f"model_seed_{MODEL_SEED}"
            / "predictions_test.csv"
        )
        if not path.exists():
            missing.append(path)
            continue
        fold_df = pd.read_csv(path)
        fold_df["fold"] = fold
        rows.append(fold_df)

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prediction files:\n{missing_text}")
    if not rows:
        raise FileNotFoundError("No prediction files found.")

    df = pd.concat(rows, ignore_index=True)
    required = {"y_true", "score"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Prediction files are missing columns: {sorted(missing_cols)}")
    return df
