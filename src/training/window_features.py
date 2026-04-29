import numpy as np


def extract_features(X, sampling_rate_hz=None):
    """Extract simple per-window features from triaxial windows.

    X is expected to have shape (n_windows, window_len, 3), with axes ordered
    as x, y, z.
    """

    X = np.asarray(X, dtype=np.float32)
    ax = X[:, :, 0]
    ay = X[:, :, 1]
    az = X[:, :, 2]

    # Per-sample vector magnitudes used by several window summaries.
    c1_svm = np.sqrt(ax**2 + ay**2 + az**2)
    c2_horizontal_svm = np.sqrt(ax**2 + az**2)
    c1_peak_idx = np.argmax(c1_svm, axis=1)

    # Peak-to-peak amplitude across the three axes.
    axis_range = X.max(axis=1) - X.min(axis=1)
    c3_peak_to_peak = np.sqrt(np.sum(axis_range**2, axis=1))

    # Jerk is the rate of acceleration change between adjacent samples.
    jerk = np.diff(X, axis=1)
    if sampling_rate_hz is not None:
        jerk = jerk * _sampling_rate_array(sampling_rate_hz, len(X))
    c7_jerk = np.sqrt(np.sum(jerk**2, axis=2))

    # Standard-deviation magnitudes over horizontal and full 3D axes.
    c8_horizontal_std = np.sqrt(ax.std(axis=1) ** 2 + az.std(axis=1) ** 2)
    c9_std_magnitude = np.sqrt(
        ax.std(axis=1) ** 2 + ay.std(axis=1) ** 2 + az.std(axis=1) ** 2
    )

    # Window-level approximation of horizontal signal magnitude area.
    c13_horizontal_sma = c2_horizontal_svm.mean(axis=1)

    # Simple impact/energy/timing summaries useful for fall detection.
    c1_peak_time = c1_peak_idx / X.shape[1]
    axis_energy_mean = np.mean(ax**2 + ay**2 + az**2, axis=1)
    posture_change_proxy = _posture_change_after_peak(c1_svm, c1_peak_idx)

    features = [
        c1_svm.mean(axis=1),
        c1_svm.max(axis=1),
        c1_svm.std(axis=1),
        c1_svm.max(axis=1) - c1_svm.min(axis=1),
        c1_peak_time,
        c2_horizontal_svm.mean(axis=1),
        c3_peak_to_peak,
        c7_jerk.mean(axis=1),
        c7_jerk.max(axis=1),
        c8_horizontal_std,
        c9_std_magnitude,
        c13_horizontal_sma,
        axis_energy_mean,
        posture_change_proxy,
    ]

    feature_names = [
        "c1_svm_mean",
        "c1_svm_max",
        "c1_svm_std",
        "c1_svm_range",
        "c1_peak_time",
        "c2_horizontal_svm_mean",
        "c3_peak_to_peak_amplitude",
        "c7_jerk_mean",
        "c7_jerk_max",
        "c8_horizontal_std_magnitude",
        "c9_std_magnitude",
        "c13_horizontal_sma",
        "axis_energy_mean",
        "posture_change_proxy",
    ]

    return np.column_stack(features).astype(np.float32), feature_names


def _posture_change_after_peak(c1_svm, peak_idx):
    changes = []

    for values, peak in zip(c1_svm, peak_idx):
        before = values[:peak] if peak > 0 else values[:1]
        after = values[peak + 1 :] if peak + 1 < len(values) else values[-1:]
        changes.append(after.mean() - before.mean())

    return np.asarray(changes, dtype=np.float32)


def _sampling_rate_array(sampling_rate_hz, n_windows):
    rates = np.asarray(sampling_rate_hz, dtype=np.float32)

    if rates.ndim == 0:
        return rates.reshape(1, 1, 1)

    if len(rates) != n_windows:
        raise ValueError("sampling_rate_hz must be a scalar or one value per window.")

    return rates.reshape(-1, 1, 1)
