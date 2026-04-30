# Project Progress

This document summarizes the current state of the fall-detection benchmark project.

## Goal

The project is building a benchmark that compares supervised classification models against time-series anomaly detection (TSAD) models on triaxial accelerometer fall-detection windows.

The benchmark principle is:

- classification models train on fall and non-fall windows
- TSAD models train on normal windows only
- both model families use the same dataset splits, windowing, labels, normalization policy, and evaluation protocol

## Data Processing

The preprocessing pipeline is implemented under:

```text
src/preprocessing/
```

Current parser modules:

- `parse_sisfall.py`
- `parse_fallalld.py`
- `parse_umafall.py`
- `parse_upfall.py`

The shared processed pipeline is implemented in:

```text
src/preprocessing/run_pipeline_2.py
```

The current flow is:

1. parse raw dataset files into interim trial pickles
2. build subject-wise train/validation/test splits
3. attach split labels to trial rows
4. generate overlapping fixed-duration windows
5. label windows using the fall-region overlap rule
6. save raw windows and metadata
7. fit normalization statistics separately for classification and TSAD
8. export normalized train/validation/test splits for each learning mode

Subject-wise splitting is already part of the processed pipeline. This is important because model evaluation should not leak subjects across train, validation, and test.

## Processed Artifact Schema

Final normalized artifacts are saved under:

```text
data/processed/<dataset>/classification/
data/processed/<dataset>/tsad/
```

Each mode contains:

```text
normalizer.npz
windows_train.npz
windows_val.npz
windows_test.npz
window_meta_train.csv
window_meta_val.csv
window_meta_test.csv
```

Each `windows_<split>.npz` contains:

```text
X: normalized accelerometer windows
y: binary window labels
```

The metadata CSV contains:

```text
window_id
subject_id
activity_id
trial_id
is_fall
split
sampling_rate_hz
start_idx
end_idx
window_label
tsad_train_eligible
```

The loader has been verified so that:

```text
windows_<split>.npz["y"] == window_meta_<split>.csv["window_label"]
```

for all current processed datasets and modes.

## Current Data Notes

The primary benchmark is defined as all datasets downsampled to 20 Hz with 2-second windows, giving raw input shape of 40 x 3


The currently processed artifacts are not all in that primary 20 Hz format yet. Current window shapes:

```text
SisFall:  400 x 3
FallAllD: 476 x 3
UMAFall:   40 x 3
UP-Fall:   36 x 3
```

--> THe current processed outputs are closer to native-rate or mixed-rate exports. 
The downsampled 20 Hz primary benchmark should be regenerated or stored under explicit output names before final benchmarking.

## Model Plan

The model notes are documented in: docs/models.md

Current classification model plan:

- feature-threshold or signal-threshold baseline
- Random Forest on engineered features
- XGBoost or SVM on engineered features
- 1D CNN classifier on raw normalized windows
- LSTM classifier on raw normalized windows

Current TSAD model plan:

- Isolation Forest on engineered features
- Dense autoencoder on flattened raw windows
- LSTM autoencoder on raw windows

Optional later TSAD extensions:

- Conv1D autoencoder
- TranAD
- Anomaly Transformer

Existing neural model architecture files are under: src/models/

Current files include:

- `cnn1d.py`
- `cnn1d_large.py`
- `lstm_classifier.py`
- `dense_ae.py`
- `lstm_ae.py`

Shallow model construction now lives directly inside the corresponding training scripts.

## Training Modules Added

Training utilities are under:  src/training/

Implemented so far:

- `data.py`
- `extract_features.py`
- `evaluation.py`
- `thresholds.py`
- `train_random_forest.py`
- `train_xgboost.py`
- `train_isolation_forest.py`
- `run_utils.py`

The shallow training scripts are documented in:

```text
docs/training_runs.md
```

### Data Loader

`src/training/data.py` loads processed train/validation/test artifacts.

Main functions:

```python
load_window_data(dataset, mode, data_root="data/processed")
make_dataloaders(data, batch_size=64, num_workers=0)
label_counts(y)
```

Example:

```python
from src.training.data import load_window_data

data = load_window_data("sisfall", "classification")
X_train = data["train"]["X"]
y_train = data["train"]["y"]
meta_train = data["train"]["meta"]
```

For neural models:

```python
from src.training.data import make_dataloaders

train_loader, val_loader, test_loader = make_dataloaders(data)
```

Python checks should be run with the project conda environment:

```text
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python
```

### Engineered Features

`src/training/extract_features.py` extracts engineered features for shallow models.

The current feature set is documented in:

```text
docs/features.md
```

Current output shape:

```text
n_windows x 13
```

Current features:

- `acc_mag_mean`
- `acc_mag_max`
- `acc_mag_std`
- `acc_mag_range`
- `peak_time`
- `horizontal_mag_mean`
- `peak_to_peak`
- `jerk_mean`
- `jerk_max`
- `horizontal_std`
- `acc_std`
- `axis_energy_mean`
- `post_peak_delta`

These features are intended for:

- Random Forest
- XGBoost or SVM
- Isolation Forest

Deep models should use the raw normalized window tensors directly.

## Training Data Handling

Classification models use:

```
data/processed/<dataset>/classification/
```

Training data contains both normal and fall windows.

TSAD models use:

```text
data/processed/<dataset>/tsad/
```

Training data contains normal windows only. Validation and test still contain both normal and fall windows so anomaly scores can be evaluated against `window_label`.

Shallow models should use engineered features:

```python
from src.training.extract_features import extract_features

X_feat, feature_names = extract_features(X, sampling_rate_hz=rates)
```

Deep models should use raw windows:

```text
X shape: n_windows x window_length x 3
```

## Evaluation Plan

Evaluation should be shared across classification and TSAD models.

Every model should produce continuous scores:

- classifier score: predicted fall probability
- TSAD score: anomaly score, where larger means more anomalous

Thresholds should be selected on the validation split only, then applied once to the test split.

Metrics to report:

- AUROC
- AUPRC
- Precision
- Recall / Sensitivity
- Specificity
- F1
- confusion matrix

Window-level metrics are the current priority. Trial-level or event-level evaluation can be added later once the window-to-event aggregation policy is stable.

## Version Control Policy

Keep source code and lightweight documentation under version control:

- `src/`
- `docs/`
- `requirements.txt`
- small hand-written config files, once experiment configs are added

Do not keep generated run artifacts in git. The repository ignores:

```text
runs/
```

Run outputs can become large and change frequently, especially for deep models. This includes:

- `model.pkl`
- PyTorch checkpoints
- TensorBoard event files
- per-window prediction CSVs
- plots generated during training/evaluation
- smoke-test run folders

For reproducibility, each run should save lightweight metadata inside its run directory:

- `config.json`
- `metrics.json`
- `feature_names.json`
- `feature_importance.csv` for shallow models, when available

Those files are useful for inspection, but they are still generated outputs and should not be committed by default. Final thesis/report tables can be copied into a small curated results document later if needed.

Current shallow training scripts use `--model-seed` for model randomness. This is separate from the preprocessing split seed used by the temporary percentage-based subject split.

Curated final or checkpoint results should be recorded in:

```text
docs/results_summary.md
```

That document should contain small result tables only, not full run artifacts.

## Next Steps

Completed shallow-model training pieces:

- Random Forest classification training script
- XGBoost classification training script
- Isolation Forest TSAD training script
- shared validation-threshold selection
- shared binary metric computation
- shared run-output utilities
- generated run output structure under `runs/benchmark/...`
- development checks on UMAFall for the three shallow scripts

Recommended next implementation steps:

1. Add a training script for the first deep classifier, likely `CNN1D`.
2. Add a shared PyTorch classification loop for CNN/LSTM classifiers.
3. Add a shared PyTorch autoencoder loop for Dense AE and LSTM AE.
4. Regenerate or clearly separate the final 20 Hz primary benchmark artifacts.
5. Add explicit experiment configs once command-line arguments become repetitive.
6. Add final split protocols, such as leave-one-subject-out or k-fold subject validation.

The next main task is to build the first complete deep-model path:

```text
load processed data -> build dataloaders -> train CNN1D -> score validation/test -> evaluate
```
