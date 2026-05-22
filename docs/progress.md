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

The completed primary benchmark uses all datasets downsampled to 20 Hz with 2-second windows and 50% overlap. This gives a raw deep-model input shape of `40 x 3`.

Primary processed dataset names:

```text
fallalld_20hz
sisfall_20hz
umafall_20hz
upfall_20hz
```

The current primary benchmark split protocol is subject-disjoint train/validation/test with split seed 7 and a target 70/15/15 ratio. Classification normalization is fit on all training windows. TSAD normalization is fit on normal training windows only, and TSAD model training uses normal training windows only.

Native-rate or alternate-window artifacts should be stored under explicit variant names, not over the primary 20 Hz outputs. Recommended naming pattern:

```text
<dataset>_native_2s
<dataset>_20hz_3s
<dataset>_20hz_5s
<dataset>_native_3s
<dataset>_easy_adl_20hz_2s
```

## Model Plan

The model notes are documented in: docs/models.md

Classification model roster:

- feature-threshold or signal-threshold baseline
- Random Forest on engineered features
- XGBoost or SVM on engineered features
- 1D CNN classifier on raw normalized windows
- LSTM classifier on raw normalized windows

TSAD model roster:

- Isolation Forest on engineered features
- Dense autoencoder on flattened raw windows
- LSTM-autoencoder 
- Conv1D autoencoder
- TranAD

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
- `deep_utils.py`
- `run_utils.py`
- `train_random_forest.py`
- `train_xgboost.py`
- `train_isolation_forest.py`
- `train_cnn1d.py`
- `train_lstm_classifier.py`
- `train_dense_ae.py`
- `train_lstm_ae.py`

The training scripts are documented in:

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

### Engineered Features

`src/training/extract_features.py` extracts engineered features for shallow models.

The current feature set is documented in `docs/features.md`


Current output shape is `n_windows x 13`. The current features are:

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

These features are intended for shallow models

## Training Data Handling

Classification models use: `data/processed/<dataset>/classification/`, where training data contains both normal and fall windows.

TSAD models use` data/processed/<dataset>/tsad/`

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

### Deep Training Utilities

`src/training/deep_utils.py` contains the shared PyTorch training code for the deep baselines.

Current shared functions:

```python
set_seed(seed)
get_device(device)
train_classifier(...)
train_autoencoder(...)
predict_classifier_scores(...)
predict_reconstruction_scores(...)
save_checkpoint(...)
```

The deep classification scripts use:

- raw normalized window tensors
- `BCEWithLogitsLoss`
- class imbalance weighting via `pos_weight = normal_count / fall_count`
- sigmoid classifier scores, where larger means more fall-like

The deep TSAD autoencoder scripts use:

- raw normalized window tensors
- normal-only TSAD training windows
- MSE reconstruction loss
- per-window mean squared reconstruction error as the anomaly score

For both families, the validation split is scored after each epoch. The best model state is selected by validation F1 after choosing the operating threshold on the validation split.

Current deep training entry points:

```text
src/training/train_cnn1d.py
src/training/train_lstm_classifier.py
src/training/train_dense_ae.py
src/training/train_lstm_ae.py
```

Deep run outputs are saved under the same benchmark structure as the shallow scripts:

```text
runs/benchmark/<dataset>/classification/<model>/model_seed_<model_seed>/
runs/benchmark/<dataset>/tsad/<model>/model_seed_<model_seed>/
```

Current deep saved files:

```text
config.json
metrics.json
predictions_val.csv
predictions_test.csv
training_history.csv
model.pt
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

## Completed Primary Benchmark

The primary 20 Hz, 2-second benchmark has a complete curated results matrix in:

```text
docs/results_summary.md
```

The completed matrix covers four datasets and seven models:

```text
4 datasets x 7 models = 28 benchmark rows
```

Datasets:

- `fallalld_20hz`
- `sisfall_20hz`
- `umafall_20hz`
- `upfall_20hz`

Classification models:

- Random Forest
- XGBoost
- CNN1D
- LSTM classifier

TSAD models:

- Isolation Forest
- Dense autoencoder
- LSTM autoencoder

Current primary-benchmark takeaways from `docs/results_summary.md`:

- Best test F1 by dataset:
  - `fallalld_20hz`: Random Forest, F1 0.904.
  - `sisfall_20hz`: Random Forest, F1 0.975.
  - `umafall_20hz`: XGBoost, F1 0.892.
  - `upfall_20hz`: XGBoost, F1 0.910.
- Supervised classification models outperform all TSAD models on F1 for every primary dataset.
- Best TSAD F1 by dataset: Dense autoencoder on `fallalld_20hz`; Isolation Forest on `sisfall_20hz`, `umafall_20hz`, and `upfall_20hz`.
- SisFall currently has the strongest overall benchmark results.

Generated run artifacts remain outside git under `runs/`. Some curated rows in `docs/results_summary.md` were copied from runs produced on another machine, so the summary is the source of truth for the completed primary result table.

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
- `training_history.csv` for deep models

Those files are useful for inspection, but they are still generated outputs and should not be committed by default. Final thesis/report tables can be copied into a small curated results document later if needed.

Current shallow training scripts use `--model-seed` for model randomness. This is separate from the preprocessing split seed used by the temporary percentage-based subject split.

Curated final or checkpoint results should be recorded in:

```text
docs/results_summary.md
```

That document should contain small result tables only, not full run artifacts.

## Next Steps

Completed training pieces:

- Random Forest classification training script
- XGBoost classification training script
- Isolation Forest TSAD training script
- CNN1D classification training script
- LSTM classification training script
- Dense autoencoder TSAD training script
- LSTM autoencoder TSAD training script
- shared PyTorch training utilities for deep classifiers and reconstruction autoencoders
- shared validation-threshold selection
- shared binary metric computation
- shared run-output utilities
- generated run output structure under `runs/benchmark/...`
- development checks on UMAFall for the three shallow scripts
- one-epoch deep smoke checks on UMAFall for CNN1D, LSTM classifier, Dense AE, and LSTM AE
- completed primary 20 Hz benchmark results for all four datasets and all seven baseline models
- curated primary benchmark table in `docs/results_summary.md`

Recommended next experiment steps:

1. Native-rate sensitivity benchmark.
   - Regenerate processed variants without `--target-sampling-rate-hz`.
   - Keep the same 2-second duration, 50% overlap, split seed, threshold policy, and metric set.
   - Report native-rate results separately from the 20 Hz primary benchmark because raw window lengths differ by dataset.
   - Prioritize rate-aware models first: Random Forest, XGBoost, Isolation Forest, CNN1D, LSTM classifier, and LSTM autoencoder.
   - Treat Dense autoencoder carefully because flattened input size changes with native sampling rate.

2. Window-size sensitivity benchmark.
   - Add 3-second and 5-second variants after the native-rate 2-second comparison is stable.
   - Recommended first pass: keep 20 Hz and create `60 x 3` and `100 x 3` raw-window inputs.
   - Use the same subject splits and evaluation policy so the comparison isolates window duration.
   - Run a selected model subset first before launching the full matrix: Random Forest, XGBoost, Isolation Forest, CNN1D, and the best-performing recurrent/autoencoder models from the primary benchmark.

3. Easy-ADL training experiment.
   - Define "easy ADL" using training-set-only criteria to avoid test-set leakage.
   - Start with simple activity-level diagnostics: acceleration magnitude max, jerk max, high-percentile magnitude, and false-positive rates from the trained primary models.
   - Select ADL activities that are consistently low-spike and not fall-like.
   - Keep validation and test sets unchanged for the main comparison. Filter only normal training windows so the experiment answers whether training on easier normal behavior changes generalization to all held-out ADLs and falls.
   - For classification, keep all fall training windows and restrict normal training windows to selected easy ADLs. For TSAD, train only on selected easy-ADL normal windows.
   - Store these as explicit variants such as `*_easy_adl_20hz_2s`, or add training-time filters that save the selected activity list into each run `config.json`.

4. Experiment management cleanup.
   - Add small experiment config files or scripts before launching large grids.
   - Standardize output names for sampling rate, window duration, ADL-filter policy, model seed, and split seed.
   - Add a compact summary-generation helper that reads `metrics.json` files and emits Markdown rows for `docs/results_summary.md`.

5. Later evaluation extensions.
   - Add trial-level or event-level metrics after the window-level sensitivity studies are complete.
   - Consider subject-fold or leave-one-subject-out validation only after the current fixed-split experiments are stable.

The current model-training path now covers:

```text
load processed data -> train model -> score validation/test -> choose validation threshold -> evaluate -> save run artifacts
```
