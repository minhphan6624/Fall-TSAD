# Current Preprocessing Pipeline

This document describes the preprocessing pipeline currently implemented under
`src/preprocessing/`. It is intended to be the source of truth for report
writing about how raw fall-detection datasets are converted into shared
window-level artifacts.

## Entry Points

Raw-to-interim parsing:

- `python src/preprocessing/parse_sisfall.py`
- `python src/preprocessing/parse_fallalld.py`
- `python src/preprocessing/parse_umafall.py`
- `python src/preprocessing/parse_upfall.py`
- `python -m src.preprocessing.run_pipeline_1 --datasets sisfall fallalld umafall upfall`

Interim-to-processed conversion:

- `python -m src.preprocessing.run_pipeline_2 --dataset <dataset>`

Subject-wise cross-validation artifact generation:

- `python -m src.preprocessing.run_cv_pipeline --dataset <dataset> --output-prefix <prefix>`

The valid dataset names are `sisfall`, `fallalld`, `umafall`, and `upfall`.

## Pipeline Overview

The current processing flow is:

1. parse raw dataset files into a common trial-level interim pickle
2. optionally resample each trial to a target sampling rate
3. build subject-wise train/validation/test assignments
4. attach split assignments to each trial
5. generate fixed-length overlapping accelerometer windows
6. label windows using the fall-impact overlap rule
7. save raw windows and labeled window metadata
8. fit z-score normalization statistics separately for each learning mode
9. export split-specific normalized windows and metadata for classification and TSAD

Datasets are processed independently. The processed output directory is
`data/processed/<output_dataset>/`, where `<output_dataset>` is either the
input dataset name or the value passed with `--output-dataset`.

## Dataset Parsing

All parsers write the same interim trial schema, but the raw extraction differs
by dataset:

| Dataset | Raw source | Accelerometer selection | Interim `acc` units | Sampling rate stored |
| --- | --- | --- | --- | --- |
| SisFall | `data/raw/sisfall/**/*.txt` | first accelerometer columns | converted to `g` with the ADXL345 conversion factor | `200` Hz |
| FallAllD | `data/raw/FallAllD/*_D3_*_A.dat` | waist device `D3` | original file/device values are preserved; no dataset-wide unit conversion is applied | `238` Hz |
| UMAFall | `data/raw/UMAFall/*.csv` | waist accelerometer only: `sensor_type == 0`, `sensor_id == 2` | original values from the selected accelerometer rows are preserved | `20` Hz |
| UP-FALL | `data/raw/UP-FALL.csv` | belt accelerometer columns grouped by subject/activity/trial | original belt accelerometer values from the CSV are preserved; these columns are already labeled in `g` by the dataset header | `18` Hz |

The interim trial schema does not store a separate unit field. The unit policy
above is therefore part of the parser contract for interpreting each trial's
`acc` array.

SisFall files whose subject folder does not match the subject encoded in the
filename are skipped. This currently handles known subject-folder/filename
mismatches, including the `SA15`/`SE15` issue in the raw data.

## Core Defaults

Default values used by `run_pipeline_2.py`:

| Setting | Default |
| --- | --- |
| `seed` | `7` |
| `split_protocol` | `default` |
| `n_folds` | `5` |
| `fold_index` | `0` |
| `window_seconds` | `2.0` |
| `overlap` | `0.50` |
| `target_sampling_rate_hz` | unset |

Window geometry is computed per trial:

```text
window_length = round(window_seconds * sampling_rate_hz)
stride = max(1, round(window_length * (1 - overlap)))
```

For the common 20 Hz, 2 second, 50% overlap setup, each window has shape
`40 x 3` and a stride of `20` samples. Trials shorter than one full window are
skipped.

## Optional Resampling

If `--target-sampling-rate-hz` is provided, every trial is resampled before
splitting and windowing. Resampling uses `scipy.signal.resample_poly` with a
rational approximation of `target_hz / source_hz`.

Upsampling is rejected unless `--allow-upsample` is set. If a trial is already
at the target rate, the signal is copied without calling SciPy.

Resampling updates:

- `acc`
- `sampling_rate_hz`
- `n_samples`

It also adds these trial-level columns:

- `source_sampling_rate_hz`
- `target_sampling_rate_hz`
- `resample_method`

These extra columns are retained in `trials_with_split.pkl`, but they are not
copied into window metadata.

## Subject Splitting

Splits are subject-disjoint: a subject appears in only one split within a
processed dataset.

### Default Split

The default split protocol:

- shuffles fall subjects and ADL-only subjects with `numpy.random.default_rng(seed)`
- reserves one fall subject for validation and one fall subject for test when available
- fills validation and test toward a `70/15/15` subject ratio
- assigns remaining subjects to training

### Subject K-Fold Split

The `subject_kfold` protocol is used for cross-validation. For fold `i`:

- fold `i` is the test split
- fold `(i + 1) % n_folds` is the validation split
- all other folds are the training split

Fall subjects are distributed round-robin across folds after shuffling. ADL-only
subjects are assigned to the currently smallest fold.

`run_cv_pipeline.py` calls `run_pipeline_2.run_pipeline()` once per fold and
writes outputs as:

```text
data/processed/<output-prefix>_fold0/
data/processed/<output-prefix>_fold1/
...
```

### Manual Split CSV

`--manual-split-csv` can override automatic splitting. The CSV must assign every
subject exactly once and include all required split names: `train`, `val`, and
`test`.

## Stage Schemas

### Stage 0: Interim Trial Pickle

Path pattern:

```text
data/interim/<dataset>/<pickle_name>.pkl
```

One row is stored per trial.

Required schema:

- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- `sampling_rate_hz`
- `n_samples`
- `acc`
- `raw_file`

Notes:

- `acc` is a NumPy array with shape `(T, 3)` and dtype compatible with `float32`
- `raw_file` is retained for provenance/debugging
- dataset identity is implied by the directory and pickle name

### Stage 1: Subject Summary

Saved as:

```text
data/processed/<output_dataset>/subject_summary.csv
```

Schema:

- `subject_id`
- `n_trials`
- `n_fall_trials`
- `n_adl_trials`
- `has_fall`

### Stage 2: Subject Splits

Saved as:

```text
data/processed/<output_dataset>/subject_splits.csv
```

Automatic split schema:

- `subject_id`
- `split`
- `split_order`
- `split_seed`
- `split_protocol`
- `fold_index`
- `fold_id`
- `n_folds`

For the default split, `split_protocol == "default"` and fold fields are `-1`.
For k-fold splits, `split_protocol == "subject_kfold"` and fold fields identify
the generated CV assignment.

Manual split files are validated and saved as provided, so their columns may be
smaller than the automatic schema.

### Stage 3: Trials With Split

Saved as:

```text
data/processed/<output_dataset>/trials_with_split.pkl
```

Base schema:

- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- `sampling_rate_hz`
- `n_samples`
- `acc`
- `raw_file`
- `split`

If resampling was enabled, the file also contains:

- `source_sampling_rate_hz`
- `target_sampling_rate_hz`
- `resample_method`

### Stage 4: Window Tensor And Pre-Label Metadata

Produced by `window_trials.generate_windows()`.

Window tensor:

```text
X_raw shape = (n_windows, window_length_samples, 3)
```

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
- `start_idx` is inclusive and `end_idx` is exclusive
- indices are sample indices within the source trial after any optional resampling
- `dataset`, `trial_row_idx`, `window_length`, `window_stride`, and `raw_file` are not stored in window metadata

### Stage 5: Labeled Window Metadata

Produced by `label_windows.label_windows()`.

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

Fall-label rule:

1. For each fall trial, compute signal magnitude vector:
   `sqrt(ax^2 + ay^2 + az^2)`.
2. Use the maximum-magnitude sample as the impact index.
3. Define the fall region as one second before to one second after the impact,
   clipped to the trial bounds.
4. Compute each window's overlap with this fall region.
5. Label a window as fall when it is from a fall trial and:
   `overlap_with_fall_region / fall_region_length > 0.60`.

Non-fall trials always produce `window_label == 0`. Windows from fall trials
outside the detected fall region can also be labeled normal.

`tsad_train_eligible` is `1` only for windows where:

```text
split == "train" and window_label == 0
```

Intermediate fall-region columns are computed and then dropped before export:

- `impact_idx`
- `fall_start`
- `fall_end`

### Stage 6: Raw Window Export

Saved as:

```text
data/processed/<output_dataset>/raw_windows/windows_all.npz
data/processed/<output_dataset>/raw_windows/window_meta_all.csv
```

Contents:

- `windows_all.npz` contains `X`, the unnormalized raw window tensor
- `window_meta_all.csv` contains the Stage 5 metadata schema

### Stage 7: Normalized Mode Exports

Saved under:

```text
data/processed/<output_dataset>/classification/
data/processed/<output_dataset>/tsad/
```

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

Each `windows_<split>.npz` contains:

- `X`: normalized window tensor
- `y`: integer `window_label` vector aligned to `X`

Split metadata schema matches the Stage 5 metadata schema after any
mode-specific filtering.

Mode-specific behavior:

- classification normalization is fit on all training windows
- TSAD normalization is fit only on training windows where `window_label == 0`
- classification train export contains fall and normal training windows
- TSAD train export contains only normal training windows
- validation and test exports are not filtered by mode

Normalization is per-axis z-score normalization. The fitted `mean` and `std`
are computed over both the window and time axes, giving one value per
accelerometer axis. Zero standard deviations are replaced with `1.0`.

## Optional Easy-ADL Filtering

`--adl-filter-config <path>` enables the easy-ADL experiment variant. The file
is a JSON mapping from dataset name to the normal training `activity_id` values
to keep.

When enabled:

- `data/processed/<output_dataset>/adl_filter.json` records the selected normal training activities
- normalization is fit after applying the training filter
- classification training keeps all fall windows and only selected normal ADL windows
- TSAD training keeps only selected normal ADL windows
- validation and test exports are unchanged

This filter affects training-set artifacts only. It does not remove validation
or test windows.
