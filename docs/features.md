# Engineered Window Features

This document describes the engineered features used by the shallow models.

These features are extracted from normalized triaxial accelerometer windows with shape:

```text
n_windows x window_length x 3
```

The three channels are treated as:

```text
ax, ay, az
```

The feature extractor is implemented in:

```text
src/training/window_features.py
```

## Feature Table

| Feature name | Definition | Why it matters |
| --- | --- | --- |
| `c1_svm_mean` | Mean of `sqrt(ax^2 + ay^2 + az^2)` over the window. | Captures average total acceleration magnitude. |
| `c1_svm_max` | Maximum of `sqrt(ax^2 + ay^2 + az^2)` over the window. | Captures impact strength, which is often high during falls. |
| `c1_svm_std` | Standard deviation of `sqrt(ax^2 + ay^2 + az^2)` over the window. | Captures how variable the total acceleration is. |
| `c1_svm_range` | `max(C1) - min(C1)` where `C1 = sqrt(ax^2 + ay^2 + az^2)`. | Captures the spread between quiet and high-impact portions of a window. |
| `c1_peak_time` | `argmax(C1) / window_length`. | Captures where the largest impact occurs inside the window. |
| `c2_horizontal_svm_mean` | Mean of `sqrt(ax^2 + az^2)` over the window. | Captures average horizontal-plane acceleration using the x/z convention from the reference feature table. |
| `c3_peak_to_peak_amplitude` | `sqrt(range_x^2 + range_y^2 + range_z^2)`, where each range is `max(axis) - min(axis)`. | Captures total peak-to-peak acceleration amplitude across all axes. |
| `c7_jerk_mean` | Mean jerk magnitude. Jerk is `diff(X) * sampling_rate_hz`, then `sqrt(jx^2 + jy^2 + jz^2)`. | Captures average rate of acceleration change. |
| `c7_jerk_max` | Maximum jerk magnitude over the window. | Captures sudden acceleration changes around impact. |
| `c8_horizontal_std_magnitude` | `sqrt(std(ax)^2 + std(az)^2)`. | Captures horizontal-plane variability. |
| `c9_std_magnitude` | `sqrt(std(ax)^2 + std(ay)^2 + std(az)^2)`. | Captures full 3D acceleration variability. |
| `c13_horizontal_sma` | Mean of `sqrt(ax^2 + az^2)` over the window. | Window-level approximation of horizontal signal magnitude area. |
| `axis_energy_mean` | Mean of `ax^2 + ay^2 + az^2` over the window. | Captures total signal energy. |
| `posture_change_proxy` | Mean `C1` after the peak minus mean `C1` before the peak. | Simple proxy for change around the impact point. |

## Notes

`c2_horizontal_svm_mean` and `c13_horizontal_sma` are currently numerically identical because both are implemented as the mean horizontal magnitude over the window. They are both kept for now to preserve the mapping to the reference feature codes. One can be removed later if duplicate features become a concern.

`c7_jerk_mean` and `c7_jerk_max` should receive `sampling_rate_hz` when extracting features. This converts acceleration change per sample into acceleration change per second.

These features are intended for feature-based shallow models:

- Random Forest
- XGBoost
- Isolation Forest

Deep models use the raw normalized window tensors directly instead of this feature table.
