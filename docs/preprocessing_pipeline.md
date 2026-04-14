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

1. Raw-to-interim parsing under `src/preprocessing/`
2. Interim-to-processed transformation for split, window, label, normalize, and export

The interim layer is dataset-specific. The processed layer should be shared across datasets as much as possible.

All dataset are processed individually. There will be no merging or bundling of datasets

## Interim Trial Schema

Each dataset is converted into a pickle under `data/interim/<dataset>/` with one row per trial.

Current shared columns:

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
- dataset identity comes from the dataset-local path and filename rather than a row-level column

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

## Common Processed Pipeline

Once a dataset has been parsed into the interim format, `run_pipeline.py` transforms it through the following stages:

1. load the interim trial pickle
2. build subject-wise train/val/test splits
3. generate fixed-length overlapping windows from each trial
4. label each window using the event-based fall rule
5. fit normalization statistics on training windows only
6. export raw and normalized windows plus companion metadata

Expected processed outputs:

- numeric arrays in `.npz`
- companion metadata in `.csv`

## Schema By Stage

This section reflects the current code in `common.py`, `build_splits.py`, `window_trials.py`, `label_windows.py`, and `run_pipeline.py`.

### Stage 0: Interim Trial Pickle

Path pattern:

- `data/interim/<dataset>/<pickle_name>.pkl`

Granularity:

- one row per trial

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

- `acc` stores the full accelerometer sequence as a NumPy array with shape `(T, 3)`
- `raw_file` is kept only at the trial level for provenance/debugging
- dataset identity is inferred from the dataset-local file path, not from a `dataset` column

### Stage 1: Subject Summary

Saved as:

- `data/processed/<dataset>/subject_summary.csv`

Granularity:

- one row per subject

Schema:

- `subject_id`
- `n_trials`
- `n_fall_trials`
- `n_adl_trials`
- `has_fall`

### Stage 2: Subject Splits

Saved as:

- `data/processed/<dataset>/subject_splits.csv`

Granularity:

- one row per subject

Schema:

- `subject_id`
- `split`
- `split_order`
- `split_seed`

### Stage 3: Trials With Split

Saved as:

- `data/processed/<dataset>/trials_with_split.pkl`

Granularity:

- one row per trial

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

Notes:

- this is the interim trial schema plus the assigned `split`

### Stage 4: Window Metadata Before Labeling

Produced by `window_trials.generate_windows()`

Granularity:

- one row per window

Window tensor:

- `X_raw` with shape `(n_windows, window_len_samples, 3)`

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
- `start_idx` and `end_idx` are sample indices inside the source trial
- `window_length`, `window_stride`, `trial_row_idx`, `dataset`, and `raw_file` are not stored in processed window metadata

### Stage 5: Labeled Window Metadata

Produced by `label_windows.label_windows()`

Granularity:

- one row per window

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
- `tsad_train_eligible` is `1` only for train-split windows labeled normal
- the fall-region internals used during labeling are compute-only and dropped before export:
  - `impact_idx`
  - `fall_start`
  - `fall_end`

### Stage 6: Raw Window Export

Saved as:

- `data/processed/<dataset>/raw_windows/windows_all.npz`
- `data/processed/<dataset>/raw_windows/window_meta_all.csv`

Contents:

- `windows_all.npz` stores raw `X`
- `window_meta_all.csv` stores the labeled window metadata schema from Stage 5

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

Normalization artifacts:

- `normalizer.npz` stores:
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
- TSAD train export additionally filters metadata and windows to normal windows only

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

## Interim EDA

The easiest place to inspect the waist accelerometer data is the interim layer, because each parser has already normalized the raw layout into the same trial-level schema.

Recommended entry point:

```bash
python3 -m src.utils.visualize_interim_trials
```

This command reads the four interim pickles under `data/interim/` and writes:

- `figures/interim_eda/interim_summary.csv`
- one example-plot PNG per dataset
- one trial-duration histogram PNG per dataset

Useful options:

```bash
python3 -m src.utils.visualize_interim_trials --datasets sisfall upfall --samples-per-class 4 --max-seconds 10
python3 -m src.utils.visualize_interim_trials --standardize
```

Interpretation notes:

- `sisfall` converts accelerometer 1 to physical units in `g`
- `upfall` belt accelerometer columns are already labeled in `g`
- `fallalld` currently preserves raw values, so absolute amplitude should not be compared directly against the other datasets
- `umafall` is parsed directly from the waist accelerometer rows without an additional dataset-wide unit conversion in the current parser

For EDA documentation, the most useful figures are usually:

- representative fall vs non-fall trial snippets for each dataset
- trial-duration distributions by label
- standardized example plots when you want to compare waveform shape rather than raw scale

## Next shared modules

After raw parsing, the next modules to build should be:

1. `build_splits.py`
2. `window_trials.py`
3. `label_windows.py`
4. `normalize_windows.py`
5. `run_pipeline.py`

## Subject-Wise Split Policy

Splits must be subject-disjoint.

Principles:

- split by subject before model training
- do not split by window first
- do not allow subject leakage across train, validation, and test
- store split assignments so TSAD and classification use the same split definitions

Propsed method: 
First need to generate subject summary table for each dataset, containing these columns
- subject_id 
- n_trials 
- n_fall_trials 
- n_adl_trials 
- has_fall 
This will be used for determining which subject goes to which split

Then we'll sort subjects into groups: 
- subjects with at least one fall trial  
- subjects with only ADL trials  

Based on this sorted list,  select train/val/test subjects so that (manually or generated based on a seed): 

- val has both fall and ADL overall  
- test has both fall and ADL overall  
- train has enough data  
- no subject appears in more than one split  

Split ratio: roughtly 70/15/15


## Windowing Policy

The shared pipeline should operate on trial-level `acc` arrays and produce fixed-length windows plus metadata.

Per-window metadata should include at least:

- `window_id`
- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- window start index
- window end index
- split

This metadata is required to trace every processed window back to its source trial.

Parameters: 2s windows, 50% overlap. 
Later will test 3s and 5s windows

THe window size will be derived using the window length in second * sampling rate

## Labeling Policy


Rules:

- define one explicit binary fall label per window
- use the same ground-truth rule when evaluating TSAD and classification
- keep the labeling rule independent from model family

Policy : Event-based labelling: For fall trials, find the impact point (peak sMV), then fall region is set to +-1s between that point, forming a 2s region. ANy window that overlaps more than 60% with the fall region is labelled as fall, otherwise normal

For ADL trials, all window is labled as normal/adl (0)

`activity_id` is currently treated as supporting metadata rather than the primary benchmark label. The core downstream target is `is_fall`, plus any future per-window label derived from the trial and window position.

## Normalization Policy

Normalization must be fit on training data only.

Recommended default:

- fit per-axis statistics on training windows
- reuse the same transform on validation and test
- keep normalization consistent across compared methods unless a method-specific deviation is explicitly justified

For TSAD, fit only on normal training windows. For classification: fit on all training windows

Two methods: RobustScaler or z-scaler. Not sure what to use. Currently leaning z-scaler.

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
- remove redundant dataset metadata and rely on dataset-local paths/filenames for dataset identity
- use nominal `20.0 Hz` for UMAFall and UP-FALL
