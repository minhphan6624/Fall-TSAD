# Project Experiment Runbook

This runbook tracks the main commands used to generate processed artifacts, train models, and aggregate experiment results.

Activate the project environment first:

```bash
conda activate fall-tsad
```

Run outputs are written under `runs/` by default. Processed datasets are written under `data/processed/`.

## Main Benchmark Protocol

Final benchmark results should use subject-wise 5-fold CV.

The default development workflow can still use a single subject-disjoint split, but those results are for debugging and iteration only.

Standard benchmark settings:

```text
n_folds=5
model_seed=42
run_root=runs/benchmark
```

Recommended processed dataset prefixes:

```text
sisfall_20hz
fallalld_20hz
umafall_20hz
upfall_20hz
```

## Preprocessing

Generate one normal development dataset:

```bash
python -m src.preprocessing.run_pipeline_2 \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --output-dataset sisfall_20hz
```

Generate one CV fold:

```bash
python -m src.preprocessing.run_pipeline_2 \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --split-protocol subject_kfold \
  --n-folds 5 \
  --fold-index 0 \
  --output-dataset sisfall_20hz_fold0
```

Generate all CV folds for one dataset:

```bash
python -m src.preprocessing.run_cv_pipeline \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --protocol subject_kfold \
  --n-folds 5 \
  --output-prefix sisfall_20hz
```

Generate all 20 Hz CV folds:

```bash
for dataset in sisfall fallalld umafall upfall; do
  python -m src.preprocessing.run_cv_pipeline \
    --dataset "$dataset" \
    --target-sampling-rate-hz 20 \
    --protocol subject_kfold \
    --n-folds 5 \
    --output-prefix "${dataset}_20hz"
done
```

Note: 20 Hz resampling requires SciPy because the resampling code uses `scipy.signal.resample_poly`.

## Training

Model entry points:

| Model | Entry point | Mode |
| --- | --- | --- |
| Random Forest | `src.training.train_random_forest` | classification |
| XGBoost | `src.training.train_xgboost` | classification |
| Isolation Forest | `src.training.train_isolation_forest` | tsad |
| CNN1D | `src.training.train_cnn1d` | classification |
| LSTM classifier | `src.training.train_lstm_classifier` | classification |
| Dense autoencoder | `src.training.train_dense_ae` | tsad |
| CNN1D autoencoder | `src.training.train_cnn1d_ae` | tsad |
| Large CNN1D autoencoder | `src.training.train_cnn1d_ae_large` | tsad |
| LSTM autoencoder | `src.training.train_lstm_ae` | tsad |

Train one model on one fold:

```bash
python -m src.training.train_random_forest \
  --dataset sisfall_20hz_fold0 \
  --model-seed 42
```

Train all current models on one fold:

```bash
dataset=sisfall_20hz_fold0

python -m src.training.train_random_forest --dataset "$dataset" --model-seed 42
python -m src.training.train_xgboost --dataset "$dataset" --model-seed 42
python -m src.training.train_isolation_forest --dataset "$dataset" --model-seed 42
python -m src.training.train_cnn1d --dataset "$dataset" --model-seed 42 --device auto
python -m src.training.train_lstm_classifier --dataset "$dataset" --model-seed 42 --device auto
python -m src.training.train_dense_ae --dataset "$dataset" --model-seed 42 --device auto
python -m src.training.train_cnn1d_ae --dataset "$dataset" --model-seed 42 --device auto
python -m src.training.train_lstm_ae --dataset "$dataset" --model-seed 42 --device auto
```

Train one model across all folds:

```bash
for fold in 0 1 2 3 4; do
  python -m src.training.train_random_forest \
    --dataset "sisfall_20hz_fold${fold}" \
    --model-seed 42
done
```

Train all current models across all folds for one dataset prefix:

```bash
prefix=sisfall_20hz

for fold in 0 1 2 3 4; do
  dataset="${prefix}_fold${fold}"

  python -m src.training.train_random_forest --dataset "$dataset" --model-seed 42
  python -m src.training.train_xgboost --dataset "$dataset" --model-seed 42
  python -m src.training.train_isolation_forest --dataset "$dataset" --model-seed 42
  python -m src.training.train_cnn1d --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_lstm_classifier --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_dense_ae --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_cnn1d_ae --dataset "$dataset" --model-seed 42 --device auto
  python -m src.training.train_lstm_ae --dataset "$dataset" --model-seed 42 --device auto
done
```

Deep scripts accept `--device auto`, `--device cpu`, or `--device cuda`. Use `--device auto` unless a specific device is required.

Check CUDA visibility before deep training:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
```

## Aggregating CV Results

Training outputs are saved as:

```text
runs/benchmark/<dataset>/<mode>/<model>/model_seed_<seed>/metrics.json
```

For CV datasets, `<dataset>` includes the fold:

```text
runs/benchmark/sisfall_20hz_fold0/classification/random_forest/model_seed_42/metrics.json
```

Aggregate one model after all folds have been trained:

```bash
python -m src.training.aggregate_cv_metrics \
  --dataset-prefix sisfall_20hz \
  --n-folds 5 \
  --mode classification \
  --model random_forest \
  --model-seed 42
```

Aggregate all current models for one dataset prefix:

```bash
prefix=sisfall_20hz

python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model random_forest --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model xgboost --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model isolation_forest --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model cnn1d --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode classification --model lstm_classifier --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model dense_ae --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model cnn1d_ae --model-seed 42
python -m src.training.aggregate_cv_metrics --dataset-prefix "$prefix" --n-folds 5 --mode tsad --model lstm_ae --model-seed 42
```

Aggregator outputs are written to:

```text
runs/benchmark/<dataset_prefix>_cv/<mode>/<model>/model_seed_<seed>/
```

Key files:

```text
cv_metrics.json
cv_metrics.csv
fold_metrics.csv
```

Use `cv_metrics.csv` for final mean/std tables. Use `fold_metrics.csv` when inspecting fold-level variation.

## Output Files

Each training run writes:

```text
config.json
metrics.json
predictions_val.csv
predictions_test.csv
```

Shallow models also write:

```text
model.pkl
feature_names.json
feature_importance.csv
```

Deep models also write:

```text
model.pt
training_history.csv
```

The `metrics.json` file contains validation and test metrics. Final benchmark aggregation uses only the `test` section.

## Smoke Checks

Use smaller settings for quick checks.

Shallow models:

```bash
python -m src.training.train_random_forest --dataset umafall_20hz_fold0 --n-estimators 5 --run-root runs/benchmark_smoke_rf
python -m src.training.train_xgboost --dataset umafall_20hz_fold0 --n-estimators 5 --run-root runs/benchmark_smoke_xgb
python -m src.training.train_isolation_forest --dataset umafall_20hz_fold0 --n-estimators 10 --run-root runs/benchmark_smoke_iforest
```

Deep models:

```bash
python -m src.training.train_cnn1d --dataset umafall_20hz_fold0 --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_cnn1d --device auto
python -m src.training.train_lstm_classifier --dataset umafall_20hz_fold0 --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_lstm_classifier --device auto
python -m src.training.train_dense_ae --dataset umafall_20hz_fold0 --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_dense_ae --device auto
python -m src.training.train_cnn1d_ae --dataset umafall_20hz_fold0 --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_cnn1d_ae --device auto
python -m src.training.train_lstm_ae --dataset umafall_20hz_fold0 --epochs 1 --batch-size 256 --patience 1 --run-root runs/benchmark_smoke_lstm_ae --device auto
```

Smoke-test outputs should not be committed.
