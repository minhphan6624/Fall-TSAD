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
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    horizontal_mag = np.sqrt(ax**2 + az**2)
    peak_idx = np.argmax(acc_mag, axis=1)

    # Peak-to-peak amplitude across the three axes.
    axis_range = X.max(axis=1) - X.min(axis=1)
    peak_to_peak = np.sqrt(np.sum(axis_range**2, axis=1))

    # Jerk is the rate of acceleration change between adjacent samples.
    jerk = np.diff(X, axis=1)
    if sampling_rate_hz is not None:
        jerk = jerk * _sampling_rate_array(sampling_rate_hz, len(X))
    jerk_mag = np.sqrt(np.sum(jerk**2, axis=2))

    # Standard-deviation magnitudes over horizontal and full 3D axes.
    horizontal_std = np.sqrt(ax.std(axis=1) ** 2 + az.std(axis=1) ** 2)
    acc_std = np.sqrt(
        ax.std(axis=1) ** 2 + ay.std(axis=1) ** 2 + az.std(axis=1) ** 2
    )

    # Simple impact/energy/timing summaries useful for fall detection.
    peak_time = peak_idx / X.shape[1]
    axis_energy_mean = np.mean(ax**2 + ay**2 + az**2, axis=1)
    post_peak_delta = _posture_change_after_peak(acc_mag, peak_idx)

    features = [
        acc_mag.mean(axis=1),
        acc_mag.max(axis=1),
        acc_mag.std(axis=1),
        acc_mag.max(axis=1) - acc_mag.min(axis=1),
        peak_time,
        horizontal_mag.mean(axis=1),
        peak_to_peak,
        jerk_mag.mean(axis=1),
        jerk_mag.max(axis=1),
        horizontal_std,
        acc_std,
        axis_energy_mean,
        post_peak_delta,
    ]

    feature_names = [
        "acc_mag_mean",
        "acc_mag_max",
        "acc_mag_std",
        "acc_mag_range",
        "peak_time",
        "horizontal_mag_mean",
        "peak_to_peak",
        "jerk_mean",
        "jerk_max",
        "horizontal_std",
        "acc_std",
        "axis_energy_mean",
        "post_peak_delta",
    ]

    return np.column_stack(features).astype(np.float32), feature_names


def _posture_change_after_peak(acc_mag, peak_idx):
    changes = []

    for values, peak in zip(acc_mag, peak_idx):
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
