def build_isolation_forest(
    n_estimators: int = 300,
    contamination: str | float = "auto",
    max_samples: str | int | float = "auto",
    max_features: int | float = 1.0,
    random_state: int | None = None,
    n_jobs: int = -1,
):
    """
    Build the Isolation Forest TSAD baseline.

    Expected TRAINING input: engineered features from normal training windows with shapes (n_windows, n_features).
    """

    from sklearn.ensemble import IsolationForest

    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def anomaly_score(model, features):
    """Return larger-is-more-anomalous Isolation Forest scores."""

    return -model.decision_function(features)


ISOLATION_FOREST_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_samples": ["auto", 0.5, 0.8],
}
