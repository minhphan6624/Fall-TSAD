from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def validate_acc(acc: np.ndarray) -> None:
    if not isinstance(acc, np.ndarray):
        raise TypeError("acc must be a numpy array")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"acc must have shape (T, 3), got {acc.shape}")
    if acc.shape[0] == 0:
        raise ValueError("acc must contain at least one sample")


def resample_acc( acc: np.ndarray, source_hz: float, target_hz: float, allow_upsample: bool = False) -> tuple[np.ndarray, str]:
    """Resample a triaxial accelerometer trial to target_hz.

    Returns the resampled array and the method used. Shape (T, 3),  dtype float32.
    """
    source_hz = float(source_hz)
    target_hz = float(target_hz)

    acc = np.asarray(acc, dtype=np.float32)
    if np.isclose(source_hz, target_hz):
        return acc.copy(), "copy"
    
    if source_hz < target_hz and not allow_upsample:
        raise ValueError(f"Upsampling is not allowed: {source_hz:g} -> {target_hz:g} Hz")

    from scipy.signal import resample_poly

    ratio = Fraction(target_hz / source_hz).limit_denominator(1000)
    resampled = resample_poly(
        acc,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
    )
    resampled = np.asarray(resampled, dtype=np.float32)

    validate_acc(resampled)

    return resampled, "scipy.signal.resample_poly"


def resample_trials_df(
    trials_df: "pd.DataFrame",
    target_hz: float,
    allow_upsample: bool = False,
):
    """Resample every trial-level acc array and update sampling metadata."""

    required = {"acc", "sampling_rate_hz", "n_samples"}
    missing = required - set(trials_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = trials_df.copy(deep=True)
    source_rates = out["sampling_rate_hz"].astype(float).tolist()
    resampled_acc = []
    methods = []

    for acc, source_hz in zip(out["acc"], source_rates):
        new_acc, method = resample_acc(
            acc=acc,
            source_hz=source_hz,
            target_hz=target_hz,
            allow_upsample=allow_upsample,
        )
        resampled_acc.append(new_acc)
        methods.append(method)

    out["acc"] = resampled_acc
    out["sampling_rate_hz"] = float(target_hz)
    out["n_samples"] = [int(acc.shape[0]) for acc in resampled_acc]
    out["source_sampling_rate_hz"] = source_rates
    out["target_sampling_rate_hz"] = float(target_hz)
    out["resample_method"] = methods
    return out
