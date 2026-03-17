# Processing Pipeline Plan

## Summary

We are rebuilding the preprocessing pipeline under `src/preprocessing1/`.

The pipeline will be split into two layers:

1. Raw-to-interim parsing
2. Interim-to-processed preprocessing

The main idea is to first convert each dataset into a clean trial-level intermediate struct, then build the shared split/window/label/normalize pipeline on top of those structs.

For this project, we are using accelerometer data only. Where the dataset provides multiple placements, we will use the waist accelerometer stream only.

## Interim Data Strategy

Each dataset gets its own parser and its own interim pickle file.

We are not merging all raw datasets into one combined raw table at this stage.

Each interim row represents one trial/recording and keeps metadata together with the accelerometer array.

Required interim schema:

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

- `acc` must be a numpy array with shape `(T, 3)`
- `n_samples` must match `acc.shape[0]`
- metadata and signal stay together in the interim struct
- processed window outputs may later separate numeric arrays from metadata

## Shared Code in `src/preprocessing1/common.py`

`common.py` is responsible only for shared schema and storage utilities.

It should contain:

- repo/data path constants
- canonical interim column order
- `validate_acc_array(...)`
- `validate_trial_row(...)`
- `make_trial_row(...)`
- `build_trial_df(...)`
- `save_interim_df(...)`

It should not contain dataset-specific parsing logic.

## Dataset Parser Plan

We will implement one parser file per dataset:

- `src/preprocessing1/parse_sisfall.py`
- `src/preprocessing1/parse_fallalld.py`
- `src/preprocessing1/parse_umafall.py`
- `src/preprocessing1/parse_upfall.py`

Each parser is responsible for:

1. Scanning raw files for that dataset
2. Extracting metadata from filenames and/or headers
3. Selecting the accelerometer stream used for this project
4. Applying dataset-specific raw parsing rules
5. Building trial rows with the shared interim schema
6. Validating those rows
7. Saving one dataset-specific pickle under `data/interim/<dataset>/`

### Parser order

Implementation order:

1. SisFall
2. FallAllD
3. UMAFall
4. UP-FALL

Reason:

- SisFall is the cleanest dataset to lock the pattern first
- FallAllD is straightforward after that
- UMAFall requires mixed-sensor demultiplexing
- UP-FALL can follow once the shared parser pattern is stable

## Dataset-Specific Notes

### SisFall

- Scan `data/raw/sisfall/**/*.txt`
- Exclude `Readme.txt`
- Parse filename metadata from:
  - activity code `Dxx` / `Fxx`
  - subject `SAxx` / `SExx`
  - trial `Rxx`
- Use only the first accelerometer stream
- Sampling rate: `200 Hz`

### FallAllD

- Parse filename metadata from trial files
- Use waist accelerometer only
- Ignore gyroscope, magnetometer, and barometer for this pipeline
- Keep one row per trial

### UMAFall

- Read header metadata from comment lines
- Skip comment/blank lines before payload
- Parse semicolon-separated data rows
- Demultiplex sensor rows using:
  - sensor type
  - sensor id / body position
- Use waist accelerometer only

### UP-FALL

- Build the same trial-level interim output
- Select the accelerometer stream intended for the project
- Keep dataset-specific parsing isolated in its parser

## Unit Conversion Decision

Where a dataset provides a reliable sensor conversion formula in its documentation, convert accelerometer readings to physical units during parsing.

Reason:

- the interim struct should already represent a clean usable signal
- downstream preprocessing should not need to know raw ADC/bit encodings
- keeping conversion at parse time makes datasets easier to compare and debug

This applies only when the conversion rule is clear and stable from the dataset documentation.

## Preprocessing Pipeline After Parsing

Once all datasets can be parsed into the interim format, the shared preprocessing stage will operate on the interim pickle rather than raw files.

Shared pipeline stages:

1. Load interim dataset pickle
2. Apply subject-wise split
3. Window the `acc` array for each trial
4. Build per-window metadata
5. Apply task-specific labeling
6. Fit normalization on training data only
7. Export processed arrays and metadata

Expected processed outputs will follow the current project convention:

- numeric arrays in `.npz`
- companion metadata in `.csv`

## Split and Windowing Principles

- split by subject, never by window first
- do not allow subject leakage across train/val/test
- parsing does not perform splitting
- parsing does not perform filtering, normalization, or windowing

## Validation and Acceptance Checks

For every parser:

- all rows must have `acc.shape[1] == 3`
- all rows must have `n_samples == acc.shape[0]`
- all rows must have valid `subject_id`, `trial_id`, `activity_id`
- `is_fall` must be derived consistently from dataset metadata
- output pickle must load successfully

For the shared pipeline later:

- splits must be subject-disjoint
- train-only statistics must be used for normalization
- processed outputs must keep metadata traceable back to the raw file

## Immediate Next Steps

1. Finalize `src/preprocessing1/common.py`
2. Implement and validate `src/preprocessing1/parse_sisfall.py`
3. Inspect the saved SisFall interim pickle
4. Implement `parse_fallalld.py`
5. Implement `parse_umafall.py`
6. Implement `parse_upfall.py`
7. Start the shared split/windowing pipeline on top of the interim structs
