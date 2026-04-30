import numpy as np
from sklearn.metrics import f1_score


def find_best_f1_threshold(y_true, scores):
    """Pick the threshold with the best validation F1."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float32)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.unique(scores):
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold
