# Common Preprocessing Pipeline

This document replaces the legacy preprocessing notes and summarizes the current preprocessing design for the benchmarking study comparing time-series anomaly detection (TSAD) and classification methods for fall detection.

## Goal

Build one shared preprocessing pipeline that:

1. Converts each raw dataset into the same trial-level interim structure
2. Builds subject-wise splits without leakage
3. Generates comparable windowed inputs for both TSAD and classification
4. Applies shared labeling, normalization, and export rules so the benchmark differs by model family rather than preprocessing

## Pipeline Structure

The pipeline is split into two layers:

1. Raw-to-interim parsing under `src/preprocessing1/`
2. Interim-to-processed transformation for split, window, label, normalize, and export

The interim layer is dataset-specific. The processed layer should be shared across datasets as much as possible.

## Interim Trial Schema

Each dataset is converted into a pickle under `data/interim/<dataset>/` with one row per trial.

Current shared columns:

- `dataset`
- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- `sampling_rate_hz`
- `n_samples`
- `acc`
- `raw_file`

Notes:

- `acc` is a NumPy array of shape `(T, 3)`
- `n_samples` must equal `acc.shape[0]`
- `is_fall` is the main task label for the benchmark
- `activity_id` is retained as metadata for provenance, inspection, and optional later grouping
- `dataset` is currently kept for traceability even though each pickle is dataset-local

## Dataset Parsing Decisions

### SisFall

- Parser: `src/preprocessing1/parse_sisfall.py`
- Raw root: `data/raw/sisfall`
- Use only the first accelerometer stream
- Convert to physical units using the documented sensor conversion
- Sampling rate: `200.0 Hz`
- Parse metadata from filename:
  - activity: `Dxx` or `Fxx`
  - subject: `SAxx` or `SExx`
  - trial: `Rxx`

### FallAllD

- Parser: `src/preprocessing1/parse_fallalld.py`
- Raw root: `data/raw/FallAllD`
- Use waist device only: `D3`
- Use accelerometer files only: `*_A.dat`
- Keep raw accelerometer values as-is for now
- Sampling rate: `238.0 Hz`
- Derive `is_fall` from activity ranges:
  - ADL: `A001` to `A044`
  - Fall: `A101` to `A135`

### UMAFall

- Parser: `src/preprocessing1/parse_umafall.py`
- Raw root: `data/raw/UMAFall`
- Parse metadata from filename
- Parse semicolon-separated payload rows after skipping comment and blank lines
- Use waist accelerometer only:
  - `sensor_type == 0`
  - `sensor_id == 2`
- Store `sampling_rate_hz = 20.0`
- Build unique `trial_id` as `<trial>_<date>_<time>` because the raw dataset reuses trial numbers
- Drop files that contain no waist accelerometer data
- Observed result from raw inspection:
  - 746 raw CSV files
  - 617 usable waist-accelerometer trials
  - 129 skipped files without waist accelerometer rows

### UP-FALL

- Parser: `src/preprocessing1/parse_upfall.py`
- Raw file: `data/raw/UP-FALL.csv`
- The file has 2 header rows and 47 columns
- Use belt accelerometer only:
  - columns `15:18`
- Group rows by `(subject_id, activity_id, trial_id)`
- Ignore the `Tag` column for interim parsing
- Store `sampling_rate_hz = 20.0`
- Derive `is_fall` from activity ids:
  - `1..5` => fall
  - `6..11` => non-fall
- Handle dataset incompleteness gracefully
- Observed result from raw inspection:
  - 559 trial groups
  - subject `8` is missing activity `11`, trials `2` and `3`

## Common Processed Pipeline

Once all four datasets exist in the interim format, the next focus is the shared processed pipeline for benchmarking.

Recommended stages:

1. Load interim trial pickle
2. Build subject-wise train/val/test splits
3. Window each `acc` trial into fixed-length overlapping windows
4. Emit per-window metadata
5. Assign task labels
6. Fit normalization on training data only
7. Export processed arrays and metadata

Expected processed outputs:

- numeric arrays in `.npz`
- companion metadata in `.csv`

## Benchmarking Priorities

The key methodological goal is fairness between TSAD and classification.

Shared components that should remain identical across paradigms:

- dataset inclusion
- subject splits
- window length and stride
- raw signal selection
- normalization strategy
- evaluation labels

Only the learning setup should differ:

- TSAD trains on normal training windows only
- Classification trains on both fall and non-fall training windows

## Next Shared Modules

After raw parsing, the next modules to build should be:

1. `build_splits.py`
2. `window_trials.py`
3. `label_windows.py`
4. `normalize_windows.py`
5. `export_processed.py`

This sequence matters because split and labeling policy determine whether the benchmark is valid.

## Subject-Wise Split Policy

Splits must be subject-disjoint.

Principles:

- split by subject before model training
- do not split by window first
- do not allow subject leakage across train, validation, and test
- store split assignments so TSAD and classification use the same split definitions

## Windowing Policy

The shared pipeline should operate on trial-level `acc` arrays and produce fixed-length windows plus metadata.

Per-window metadata should include at least:

- `dataset`
- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- window start index
- window end index
- split

This metadata is required to trace every processed window back to its source trial.

## Labeling Policy

For the benchmarking study, the labeling rule is one of the most important decisions.

Requirements:

- define one explicit binary fall label per window
- use the same ground-truth rule when evaluating TSAD and classification
- keep the labeling rule independent from model family

`activity_id` is currently treated as supporting metadata rather than the primary benchmark label. The core downstream target is `is_fall`, plus any future per-window label derived from the trial and window position.

## Normalization Policy

Normalization must be fit on training data only.

Recommended default:

- fit per-axis statistics on training windows
- reuse the same transform on validation and test
- keep normalization consistent across compared methods unless a method-specific deviation is explicitly justified

For TSAD:

- fit only on normal training windows

For classification:

- fit on all training windows

## Validation Checks

### Interim validation

For every dataset parser:

- `acc.shape[1] == 3`
- `n_samples == acc.shape[0]`
- `subject_id`, `trial_id`, and `activity_id` are populated
- `is_fall` is derived consistently from dataset metadata
- saved pickle loads successfully

### Processed validation

For the shared processed pipeline:

- splits are subject-disjoint
- train-only statistics are used for normalization
- output arrays and metadata have matching lengths
- each processed window is traceable back to raw trial metadata
- TSAD and classification consume the same split and window definitions

## Open Design Defaults

Current defaults locked from discussion:

- accelerometer only for all datasets
- waist-equivalent placement when multiple sensors exist
- keep `activity_id` as source metadata rather than forcing a normalized enumeration now
- keep `dataset` in the interim schema for traceability unless size measurements later show a real need to remove it
- use nominal `20.0 Hz` for UMAFall and UP-FALL

## Immediate Next Step

The next implementation target should be the shared split and window pipeline on top of the interim pickles, because that is the layer that determines benchmark comparability between TSAD and classification.
