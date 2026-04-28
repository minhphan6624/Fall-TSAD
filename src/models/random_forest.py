def build_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | float = "sqrt",
    class_weight: str | dict | None = "balanced",
    random_state: int | None = None,
    n_jobs: int = -1,
):
    """Build the primary-benchmark Random Forest classifier.

    Expected input: engineered feature matrix with shape (n_windows, n_features).
    """

    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5],
}
