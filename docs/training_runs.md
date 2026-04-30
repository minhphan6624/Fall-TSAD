# Training Run Scripts

This document records the current model training entry points and common commands.

Use the project conda environment when running Python:

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python
```

Run outputs are written under `runs/` by default. That directory is ignored by git.

## Common Output Structure

Shallow classification models write to:

```text
runs/benchmark/<dataset>/classification/<model>/model_seed_<model_seed>/
```

Shallow TSAD models write to:

```text
runs/benchmark/<dataset>/tsad/<model>/model_seed_<model_seed>/
```

Current saved files include:

- `config.json`
- `feature_names.json`
- `metrics.json`
- `predictions_val.csv`
- `predictions_test.csv`
- `model.pkl`
- `feature_importance.csv` for Random Forest and XGBoost

## Random Forest Classification

Entry point:

```text
src/training/train_random_forest.py
```

Command:

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python -m src.training.train_random_forest \
  --dataset sisfall \
  --model-seed 42
```

Useful options:

```bash
--n-estimators 300
--max-depth 10
--min-samples-leaf 2
--data-root data/processed
--run-root runs/benchmark
```

Data mode:

```text
classification
```

Input:

```text
engineered features from normalized windows
```

Score:

```text
predict_proba(X)[:, 1]
```

## XGBoost Classification

Entry point:

```text
src/training/train_xgboost.py
```

Command:

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python -m src.training.train_xgboost \
  --dataset sisfall \
  --model-seed 42
```

Useful options:

```bash
--n-estimators 300
--max-depth 3
--learning-rate 0.05
--subsample 0.8
--colsample-bytree 0.8
--data-root data/processed
--run-root runs/benchmark
```

Data mode:

```text
classification
```

Input:

```text
engineered features from normalized windows
```

Score:

```text
predict_proba(X)[:, 1]
```

## Isolation Forest TSAD

Entry point:

```text
src/training/train_isolation_forest.py
```

Command:

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python -m src.training.train_isolation_forest \
  --dataset sisfall \
  --model-seed 42
```

Useful options:

```bash
--n-estimators 300
--contamination auto
--max-samples auto
--max-features 1.0
--data-root data/processed
--run-root runs/benchmark
```

Data mode:

```text
tsad
```

Input:

```text
engineered features from normalized windows
```

Score:

```text
-model.decision_function(X)
```

Higher scores mean more anomalous or more fall-like.

## Current Shared Evaluation Behavior

All current shallow scripts:

1. load processed train/validation/test splits
2. extract engineered features using `src/training/extract_features.py`
3. train the model
4. produce continuous validation and test scores
5. choose the operating threshold by best validation F1
6. evaluate validation and test metrics
7. save model, metrics, predictions, and config under `runs/`

Metrics are computed by:

```text
src/training/evaluation.py
```

Threshold selection is implemented in:

```text
src/training/thresholds.py
```

## Smoke-Test Commands

Use small estimator counts for quick checks:

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python -m src.training.train_random_forest \
  --dataset umafall \
  --n-estimators 5 \
  --run-root runs/benchmark_smoke_rf
```

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python -m src.training.train_xgboost \
  --dataset umafall \
  --n-estimators 5 \
  --run-root runs/benchmark_smoke_xgb
```

```bash
/home/minhqphan/miniconda3/envs/fall-tsad/bin/python -m src.training.train_isolation_forest \
  --dataset umafall \
  --n-estimators 10 \
  --run-root runs/benchmark_smoke_iforest
```

Smoke-test outputs should not be committed.

## Planned Future Scripts

The next training scripts should cover deep models:

- CNN1D classifier
- LSTM classifier
- Dense autoencoder
- LSTM autoencoder

Those should reuse the same evaluation and run-output conventions where possible.
