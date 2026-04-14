# Legacy Preprocessing Notes

This document preserves earlier preprocessing plans, design discussions, and benchmark-policy notes. It is not the source of truth for the current implementation. For the active pipeline and emitted schemas, see [preprocessing_pipeline.md](/home/minhqphan/projects/Fall-TSAD/docs/preprocessing_pipeline.md).

## Original Goal

Build one shared preprocessing pipeline that:

1. Converts each raw dataset into the same trial-level interim structure
2. Builds subject-wise splits without leakage
3. Generates comparable windowed inputs for both TSAD and classification
4. Applies shared labeling, normalization, and export rules so the benchmark differs by model family rather than preprocessing

## Historical Structure

The pipeline was originally discussed as two layers:

1. raw-to-interim parsing
2. interim-to-processed transformation for split, window, label, normalize, and export

The interim layer is dataset-specific. The processed layer was intended to be shared across datasets as much as possible.

## Dataset Parsing Decisions

### SisFall

- Parser: `src/preprocessing/parse_sisfall.py`
- Raw root: `data/raw/sisfall`
- Use only the first accelerometer stream
- Convert to physical units using the documented sensor conversion
- Sampling rate: `200.0 Hz`
- Parse metadata from filename:
  - activity: `Dxx` or `Fxx`
  - subject: `SAxx` or `SExx`
  - trial: `Rxx`

### FallAllD

- Parser: `src/preprocessing/parse_fallalld.py`
- Raw root: `data/raw/FallAllD`
- Use waist device only: `D3`
- Use accelerometer files only: `*_A.dat`
- Keep raw accelerometer values as-is for now
- Sampling rate: `238.0 Hz`
- Derive `is_fall` from activity ranges:
  - ADL: `A001` to `A044`
  - Fall: `A101` to `A135`

### UMAFall

- Parser: `src/preprocessing/parse_umafall.py`
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

- Parser: `src/preprocessing/parse_upfall.py`
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

## Benchmarking Priorities

The key methodological goal was fairness between TSAD and classification.

Shared components intended to remain identical across paradigms:

- dataset inclusion
- subject splits
- window length and stride
- raw signal selection
- normalization strategy
- evaluation labels

Only the learning setup should differ:

- TSAD trains on normal training windows only
- classification trains on both fall and non-fall training windows

## Interim EDA Notes

The interim layer was identified as the easiest place to inspect the waist accelerometer data because each parser normalizes the raw layout into the same trial-level schema.

Recommended entry point:

```bash
python3 -m src.utils.visualize_interim_trials
```

Useful options:

```bash
python3 -m src.utils.visualize_interim_trials --datasets sisfall upfall --samples-per-class 4 --max-seconds 10
python3 -m src.utils.visualize_interim_trials --standardize
```

Interpretation notes:

- `sisfall` converts accelerometer 1 to physical units in `g`
- `upfall` belt accelerometer columns are already labeled in `g`
- `fallalld` preserves raw values, so absolute amplitude should not be compared directly against the other datasets
- `umafall` is parsed directly from the waist accelerometer rows without an additional dataset-wide unit conversion

## Historical Module Order

The originally proposed shared module sequence was:

1. `build_splits.py`
2. `window_trials.py`
3. `label_windows.py`
4. `normalize_windows.py`
5. `run_pipeline.py`

## Historical Policy Notes

### Subject-Wise Split Policy

- split by subject before model training
- do not split by window first
- do not allow subject leakage across train, validation, and test
- store split assignments so TSAD and classification use the same split definitions
- target a split ratio of roughly `70/15/15`

### Windowing Policy

- operate on trial-level `acc` arrays
- use fixed-length windows with overlap
- baseline parameters discussed:
  - `2 s` windows
  - `50%` overlap
- later ablations discussed:
  - `3 s`
  - `5 s`

### Labeling Policy

- define one explicit binary fall label per window
- use the same ground-truth rule for TSAD and classification
- keep the labeling rule independent from model family
- event-based labeling rule discussed:
  - find impact point using peak SMV
  - define fall region as `±1 s` around the impact
  - label a window as fall if overlap with the fall region exceeds `60%`

### Normalization Policy

- fit normalization on training data only
- TSAD: fit on normal training windows only
- classification: fit on all training windows
- z-score normalization was the leading default under discussion

## Historical Validation Checklist

Interim-level checks that were discussed:

- `acc.shape[1] == 3`
- `n_samples == acc.shape[0]`
- `subject_id`, `trial_id`, and `activity_id` are populated
- `is_fall` is derived consistently from dataset metadata
- saved pickle loads successfully

Processed-level checks that were discussed:

- splits are subject-disjoint
- train-only statistics are used for normalization
- output arrays and metadata have matching lengths
- each processed window is traceable back to source trial metadata
- TSAD and classification consume the same split and window definitions

## Historical Defaults

- accelerometer only for all datasets
- waist-equivalent placement when multiple sensors exist
- keep `activity_id` as source metadata rather than forcing a normalized enumeration
- use nominal `20.0 Hz` for UMAFall and UP-FALL
