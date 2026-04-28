def build_xgboost_classifier(
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    random_state: int | None = None,
    n_jobs: int = -1,
):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


XGBOOST_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
