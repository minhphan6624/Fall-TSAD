import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def _safe_score(metric_fn, y_true, scores):
    try:
        return float(metric_fn(y_true, scores))
    except ValueError:
        return None

def compute_binary_metrics(y_true, scores, threshold):
    """Compute binary metrics from continuous fall/anomaly scores."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "auroc": _safe_score(roc_auc_score, y_true, scores),
        "auprc": _safe_score(average_precision_score, y_true, scores),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
