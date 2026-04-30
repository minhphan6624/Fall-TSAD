# Engineered Window Features

This document describes the engineered features used by the shallow models.

These features are extracted from normalized triaxial accelerometer windows with shape:

```text
n_windows x window_length x 3
```

The three channels are treated as: ax, ay, az


The feature extractor is implemented in: src/training/extract_features.py

## Feature Table

| Feature name | Definition | Why it matters |
| --- | --- | --- |
| `acc_mag_mean` | Mean of `sqrt(ax^2 + ay^2 + az^2)` over the window. | Captures average total acceleration magnitude. |
| `acc_mag_max` | Maximum of `sqrt(ax^2 + ay^2 + az^2)` over the window. | Captures impact strength, which is often high during falls. |
| `acc_mag_std` | Standard deviation of `sqrt(ax^2 + ay^2 + az^2)` over the window. | Captures how variable the total acceleration is. |
| `acc_mag_range` | `max(acc_mag) - min(acc_mag)` where `acc_mag = sqrt(ax^2 + ay^2 + az^2)`. | Captures the spread between quiet and high-impact portions of a window. |
| `peak_time` | `argmax(acc_mag) / window_length`. | Captures where the largest impact occurs inside the window. |
| `horizontal_mag_mean` | Mean of `sqrt(ax^2 + az^2)` over the window. | Captures average horizontal-plane acceleration using the x/z convention from the reference feature table. |
| `peak_to_peak` | `sqrt(range_x^2 + range_y^2 + range_z^2)`, where each range is `max(axis) - min(axis)`. | Captures total peak-to-peak acceleration amplitude across all axes. |
| `jerk_mean` | Mean jerk magnitude. Jerk is `diff(X) * sampling_rate_hz`, then `sqrt(jx^2 + jy^2 + jz^2)`. | Captures average rate of acceleration change. |
| `jerk_max` | Maximum jerk magnitude over the window. | Captures sudden acceleration changes around impact. |
| `horizontal_std` | `sqrt(std(ax)^2 + std(az)^2)`. | Captures horizontal-plane variability. |
| `acc_std` | `sqrt(std(ax)^2 + std(ay)^2 + std(az)^2)`. | Captures full 3D acceleration variability. |
| `axis_energy_mean` | Mean of `ax^2 + ay^2 + az^2` over the window. | Captures total signal energy. |
| `post_peak_delta` | Mean `acc_mag` after the peak minus mean `acc_mag` before the peak. | Simple proxy for change around the impact point. |

## Notes

`jerk_mean` and `jerk_max` should receive `sampling_rate_hz` when extracting features. This converts acceleration change per sample into acceleration change per second.

These features are intended for feature-based shallow models:

- Random Forest
- XGBoost
- Isolation Forest

Deep models use the raw normalized window tensors directly instead of this feature table.
