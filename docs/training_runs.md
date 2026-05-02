# Training Run Scripts

This document records the current model training entry points and common commands.

Use the project conda environment before running training commands:

```bash
conda activate fall-tsad
```

Run outputs are written under `runs/` by default. That directory is ignored by git.

## Output Structure

Classification models write to `runs/benchmark/<dataset>/classification/<model>/model_seed_<model_seed>/`.

TSAD models write to `runs/benchmark/<dataset>/tsad/<model>/model_seed_<model_seed>/`.

Shallow runs save `config.json`, `feature_names.json`, `metrics.json`, `predictions_val.csv`, `predictions_test.csv`, `model.pkl`, and `feature_importance.csv` for Random Forest and XGBoost.

Deep runs save `config.json`, `metrics.json`, `predictions_val.csv`, `predictions_test.csv`, `training_history.csv`, and `model.pt`.

Deep scripts accept `--device auto`, `--device cpu`, or `--device cuda`. Use `--device auto` by default. It uses CUDA when PyTorch can see a compatible GPU and falls back to CPU otherwise.

Check CUDA visibility before deep training with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
```

Make sure to check the CUDA setup on the machine that is intended to run the training, then install the corresponding pytorch dependency

## Training Commands

### Random Forest Classification

Entry point: `src/training/train_random_forest.py`

```bash
python -m src.training.train_random_forest --dataset sisfall --model-seed 42
```

Mode: `classification`. Input: engineered features from normalized windows. Score: `predict_proba(X)[:, 1]`.

### XGBoost Classification

Entry point: `src/training/train_xgboost.py`

```bash
python -m src.training.train_xgboost --dataset sisfall --model-seed 42
```

Mode: `classification`. Input: engineered features from normalized windows. Score: `predict_proba(X)[:, 1]`.

### Isolation Forest TSAD

Entry point: `src/training/train_isolation_forest.py`

```bash
python -m src.training.train_isolation_forest --dataset sisfall --model-seed 42
```

Mode: `tsad`. Input: engineered features from normalized windows. Score: `-model.decision_function(X)`. Higher scores mean more anomalous or more fall-like.

### CNN1D Classification

Entry point: `src/training/train_cnn1d.py`

```bash
python -m src.training.train_cnn1d --dataset sisfall --model-seed 42 --device auto
```

Mode: `classification`. Input: raw normalized windows with shape `n_windows x window_length x 3`. Score: `sigmoid(model(X))`. Higher scores mean more fall-like.

### LSTM Classification

Entry point: `src/training/train_lstm_classifier.py`

```bash
python -m src.training.train_lstm_classifier --dataset sisfall --model-seed 42 --device auto
```

Mode: `classification`. Input: raw normalized windows with shape `n_windows x window_length x 3`. Score: `sigmoid(model(X))`. Higher scores mean more fall-like.

### Dense Autoencoder TSAD

Entry point: `src/training/train_dense_ae.py`

```bash
python -m src.training.train_dense_ae --dataset sisfall --model-seed 42 --device auto
```

Mode: `tsad`. Input: raw normalized windows, flattened internally. Score: mean squared reconstruction error over time and axes. Higher scores mean more anomalous or more fall-like.

### LSTM Autoencoder TSAD

Entry point: `src/training/train_lstm_ae.py`

```bash
python -m src.training.train_lstm_ae --dataset sisfall --model-seed 42 --device auto
```

Mode: `tsad`. Input: raw normalized windows with shape `n_windows x window_length x 3`. Score: mean squared reconstruction error over time and axes. Higher scores mean more anomalous or more fall-like.

## Shared Behavior

All current training scripts:

1. load processed train/validation/test splits
2. prepare model inputs
3. train the model
4. produce continuous validation and test scores
5. choose the operating threshold by best validation F1
6. evaluate validation and test metrics
7. save model, metrics, predictions, and config under `runs/`

Shallow scripts prepare model inputs by extracting engineered features with `src/training/extract_features.py`.

Deep scripts use raw normalized windows directly through `src/training/data.py`.

Metrics are computed by `src/training/evaluation.py`. Threshold selection is implemented in `src/training/thresholds.py`.

Shared run-output helpers are implemented in `src/training/run_utils.py`. These handle feature extraction for a split, run-directory construction, JSON output, prediction CSV output, and shallow feature-importance output.

Deep training helpers are implemented in `src/training/deep_utils.py`. These handle seed setup, device selection, classifier training with `BCEWithLogitsLoss`, autoencoder training with MSE reconstruction loss, validation-F1 model selection, score prediction, and PyTorch checkpoint saving.

## Smoke Tests

Use small estimator counts for shallow smoke checks:

```bash
python -m src.training.train_random_forest --dataset umafall --n-estimators 5 --run-root runs/benchmark_smoke_rf
python -m src.training.train_xgboost --dataset umafall --n-estimators 5 --run-root runs/benchmark_smoke_xgb
python -m src.training.train_isolation_forest --dataset umafall --n-estimators 10 --run-root runs/benchmark_smoke_iforest
```

Use one epoch for deep smoke checks:

```bash
python -m src.training.train_cnn1d --dataset umafall --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_cnn1d --device auto
python -m src.training.train_lstm_classifier --dataset umafall --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_lstm_classifier --device auto
python -m src.training.train_dense_ae --dataset umafall --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_dense_ae --device auto
python -m src.training.train_lstm_ae --dataset umafall --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_lstm_ae --device auto
```

Smoke-test outputs should not be committed.

## Verification

The shallow scripts were checked on UMAFall with 50 estimators and `model_seed=42`.

The deep scripts were checked on UMAFall with one epoch, `batch_size=256`, and `model_seed=42`.

Checks performed:

- expected files were created
- `window_label == y_true` in prediction CSVs
- `pred_label` contained only binary labels
- scores had no missing values
- PyTorch checkpoints were saved as `model.pt`

These are development checks only. The generated folders are ignored by git.
