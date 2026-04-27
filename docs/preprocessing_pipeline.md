# Current Preprocessing Pipeline

This document describes the preprocessing pipeline as it is currently implemented in `src/preprocessing/`.

## Entry Points

Raw-to-interim parsers:

- `src/preprocessing/parse_sisfall.py`
- `src/preprocessing/parse_fallalld.py`
- `src/preprocessing/parse_umafall.py`
- `src/preprocessing/parse_upfall.py`

Raw-to-interim all-dataset entrypoint:

- `python3 -m src.preprocessing.parse_all`

Shared interim-to-processed entrypoint:

- `python3 -m src.preprocessing.run_pipeline --dataset <dataset>`

## Pipeline Overview

The current preprocessing flow is:

1. parse raw data into an interim trial pickle
2. build subject-wise train/val/test splits
3. attach split assignments to each trial
4. generate fixed-length overlapping windows
5. label windows with the event-based fall rule
6. save raw windows and metadata
7. fit normalization statistics per mode
8. export split-specific normalized windows and metadata

Datasets are processed individually. Dataset identity is inferred from the dataset-local path and filename rather than a `dataset` column in every row.

## Schema By Stage

### Stage 0: Interim Trial Pickle

Path pattern: `data/interim/<dataset>/<pickle_name>.pkl`, one row per trial

Schema:

- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- `sampling_rate_hz`
- `n_samples`
- `acc`
- `raw_file`

Notes:

- `acc` is a NumPy array with shape `(T, 3)`
- `raw_file` is retained at the trial level for provenance/debugging

### Stage 1: Subject Summary

Saved as: `data/processed/<dataset>/subject_summary.csv`, one row per subject

Schema:

- `subject_id`
- `n_trials`
- `n_fall_trials`
- `n_adl_trials`
- `has_fall`

### Stage 2: Subject Splits

Saved as `data/processed/<dataset>/subject_splits.csv`, one row per subject

Schema:

- `subject_id`
- `split`
- `split_order`
- `split_seed`

### Stage 3: Trials With Split

Saved as: `data/processed/<dataset>/trials_with_split.pkl`, one row per trial

Schema:

- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- `sampling_rate_hz`
- `n_samples`
- `acc`
- `raw_file`
- `split`

### Stage 4: Window Tensor + Pre-Label Metadata

Produced by `window_trials.generate_windows()`, one row per window

Window tensor: `X_raw` with shape `(n_windows, window_len_samples, 3)`

Metadata schema:

- `window_id`
- `subject_id`
- `activity_id`
- `trial_id`
- `is_fall`
- `split`
- `sampling_rate_hz`
- `start_idx`
- `end_idx`

Notes:

- `window_id` is the alignment key between metadata rows and `X_raw`
- `start_idx` and `end_idx` are sample indices within the source trial
- `dataset`, `trial_row_idx`, `window_length`, `window_stride`, and `raw_file` are not stored in processed window metadata

### Stage 5: Labeled Window Metadata

Produced by `label_windows.label_windows()`, one row per window

Schema:

- `window_id`
- `subject_id`
- `activity_id`
- `trial_id`
- `is_fall`
- `split`
- `sampling_rate_hz`
- `start_idx`
- `end_idx`
- `window_label`
- `tsad_train_eligible`

Notes:

- `window_label` is the final binary per-window target
- `tsad_train_eligible` is `1` only for train windows labeled normal
- fall-region intermediates are compute-only and dropped before export:
  - `impact_idx`
  - `fall_start`
  - `fall_end`

### Stage 6: Raw Window Export

Saved as:

- `data/processed/<dataset>/raw_windows/windows_all.npz`
- `data/processed/<dataset>/raw_windows/window_meta_all.csv`

Contents:

- `windows_all.npz` stores raw `X`
- `window_meta_all.csv` stores the Stage 5 metadata schema

### Stage 7: Normalized Mode Exports

Saved under:

- `data/processed/<dataset>/classification/`
- `data/processed/<dataset>/tsad/`

Files per mode:

- `normalizer.npz`
- `windows_train.npz`
- `windows_val.npz`
- `windows_test.npz`
- `window_meta_train.csv`
- `window_meta_val.csv`
- `window_meta_test.csv`

`normalizer.npz` schema:

- `mean`
- `std`
- `mode`

Split metadata schema:

- `window_id`
- `subject_id`
- `activity_id`
- `trial_id`
- `is_fall`
- `split`
- `sampling_rate_hz`
- `start_idx`
- `end_idx`
- `window_label`
- `tsad_train_eligible`

Mode-specific behavior:

- classification normalization is fit on all training windows
- TSAD normalization is fit only on training windows where `window_label == 0`
- TSAD train exports additionally filter metadata and windows to normal windows only


# Note:

SA15 D017 R01-R05 is mislabelled to SE15, whcih causes the
THis can be due to dataset poor quality. For now, they are treated as raw-data anomalies.
Later on, consider renaming it to the actual subject ID
