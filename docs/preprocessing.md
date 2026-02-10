# SisFall v2 Preprocessing Pipeline

This document describes the current preprocessing flow implemented under `src/preprocessing/sisfall`.

## Goal

Build split-aware, windowed acceleration data from raw SisFall trials using one accelerometer stream (ADXL345), ready for downstream TSAD and classification labeling/normalization.

## Scope (current)

Implemented through impact-aware window labeling:

1. File indexing and metadata extraction
2. Subject-wise split assignment
3. Signal loading and unit conversion
4. Low-pass filtering
5. Sliding-window segmentation
6. Impact-aware labeling for classification and TSAD
7. Persisting per-split windows and window metadata

## Data Source

- Raw root: `data/raw/sisfall`
- File pattern: `<ACTIVITY>_<SUBJECT>_<TRIAL>.txt`
  - Example: `D01_SA01_R01.txt`

## Processing Steps

### 1. Build Index

Module: `src/preprocessing/sisfall/build_index.py`

- Scans all SisFall trial files (`*.txt`, excluding `Readme.txt`)
- Parses filename metadata:
  - `activity`, `subject`, `group`, `trial`, `is_fall`
- Adds:
  - `path` (absolute/relative file path in workspace)
  - `n_samples` (line count)
- Writes:
  - `data/interim/sisfall/index.csv`

### 2. Build Splits

Module: `src/preprocessing/sisfall/build_splits.py`

Subject-wise split policy:

- `train`: `SA01` to `SA15`
- `val`: `SA16` to `SA17`
- `test`: remaining `SA` + all `SE`

Output:

- `data/interim/sisfall/splits.csv`

### 3. Load and Convert Signal

Module: `src/preprocessing/sisfall/load_signal.py`

- Reads each trial CSV-like text row
- Cleans trailing `;` in last column
- Converts to numeric array
- Uses only first accelerometer channels (`0:3`, ADXL345)
- Converts bits to `g` using SisFall formula:

`acc_g = ((2 * range_g) / (2 ** resolution_bits)) * raw_bits`

with ADXL345 settings:

- range: `+-16g`
- resolution: `13 bits`

Returned key:

- `{"acc1": np.ndarray[T, 3]}`

### 4. Filtering

Module: `src/preprocessing/sisfall/filtering.py`

- Applies a 4th-order Butterworth low-pass filter
- Cutoff frequency: `5 Hz`
- Sampling frequency: `200 Hz`
- Uses `filtfilt` for zero-phase filtering

### 5. Windowing

Module: `src/preprocessing/sisfall/windowing.py`

Constants:

- `FS_HZ = 200`
- `WINDOW_SECONDS = 1.0`
- `OVERLAP = 0.5`

Derived:

- `window_size = 200` samples
- `stride = 100` samples

Behavior:

- Generates sliding windows over filtered signal
- Drops files shorter than one full window
- Returns:
  - windows tensor `X` with shape `(N, 200, 3)`
  - start indices per window

Window metadata schema (lean):

- `window_local_idx`
- `file_path`
- `subject`
- `activity`
- `trial`
- `is_fall`
- `split`
- `start_idx`
- `end_idx`

### 6. Impact-Aware Labeling

Modules:

- `src/preprocessing/sisfall/labeling.py`
- integrated in `src/preprocessing/sisfall/pipeline.py`

Method:

- Impact point per file = index of maximum acceleration magnitude
  - `impact_idx = argmax(sqrt(ax^2 + ay^2 + az^2))`
- Impact region is centered around impact with `0.5s` half-width
  - region: `[impact_idx - 0.5s, impact_idx + 0.5s]`
- Window is labeled positive if overlap with region is at least `30%`

Output labels:

- `label_cls`: classification label (`1`=fall-impact window, `0`=non-fall window)
- `label_tsad`: TSAD anomaly label (`1`=anomalous impact window, `0`=normal window)
- `impact_index`: impact sample index for the source file (`-1` for ADL files)
- `tsad_train_eligible`: `1` only for ADL windows in train split (strict TSAD normal-only training mask)

### 6. Pipeline Orchestration

Module: `src/preprocessing/sisfall/pipeline.py`

`run_pipeline(...)` executes all previous steps and writes split artifacts.

Per split outputs:

- `data/interim/sisfall/windows_train.npz`
- `data/interim/sisfall/windows_val.npz`
- `data/interim/sisfall/windows_test.npz`

Each `.npz` contains:

- `X`: filtered windows, shape `(N, 200, 3)`

Metadata outputs:

- `data/interim/sisfall/window_meta_train.csv`
- `data/interim/sisfall/window_meta_val.csv`
- `data/interim/sisfall/window_meta_test.csv`

Metadata includes:

- base provenance fields (`file_path`, `subject`, `activity`, `trial`, `split`, window indices)
- impact-aware labels (`label_cls`, `label_tsad`, `impact_index`, `tsad_train_eligible`)

Also produced:

- `data/interim/sisfall/index.csv`
- `data/interim/sisfall/splits.csv`

## Design Notes

- Splitting is file-level and subject-wise before any model-specific processing to avoid leakage.
- Window metadata intentionally duplicates key identifiers from file metadata so downstream training/evaluation can run without repeated joins.
- Current scope is feature-ready signal windows only; model-specific labeling and normalization are next-stage modules.

## Next Stage (planned)

1. Branch-specific normalization fit on train split only
2. Final processed exports for classification and TSAD loaders
3. Validation scripts for leakage and label distribution checks
