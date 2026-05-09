# Subject-Wise K-Fold CV Evaluation

This project now uses subject-wise 5-fold cross-validation as the final
benchmark protocol. The older single subject-disjoint split remains useful for
development and debugging, but final reported results should come from the CV
workflow.

## Protocol

Each fold is built from complete subjects, not windows or trials.

For fold `i`:

- fold `i` is the test split
- fold `(i + 1) % n_folds` is the validation split
- all other folds are the training split

With `n_folds=5`, every subject appears in the test split exactly once across
the full CV run.

The evaluation rule is:

- train model parameters using only the training split
- choose thresholds and hyperparameters using only the validation split
- report final metrics using only the test split

This applies to both classification and TSAD models.

## Implemented Entry Points

Generate one fold directly:

```bash
python -m src.preprocessing.run_pipeline_2 \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --split-protocol subject_kfold \
  --n-folds 5 \
  --fold-index 0 \
  --output-dataset sisfall_20hz_fold0
```

Generate all folds:

```bash
python -m src.preprocessing.run_cv_pipeline \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --protocol subject_kfold \
  --n-folds 5 \
  --output-prefix sisfall_20hz
```

This creates:

```text
data/processed/sisfall_20hz_fold0/
data/processed/sisfall_20hz_fold1/
data/processed/sisfall_20hz_fold2/
data/processed/sisfall_20hz_fold3/
data/processed/sisfall_20hz_fold4/
```

Each fold directory has the same structure as the normal processed datasets:

```text
classification/
tsad/
raw_windows/
subject_summary.csv
subject_splits.csv
trials_with_split.pkl
```

Training scripts do not need special CV arguments. Use the fold dataset name:

```bash
python -m src.training.train_random_forest \
  --dataset sisfall_20hz_fold0 \
  --model-seed 42
```

## Split Metadata

Each generated `subject_splits.csv` includes:

- `subject_id`
- `split`
- `split_order`
- `split_seed`
- `split_protocol`
- `fold_index`
- `fold_id`
- `n_folds`

For a 5-fold run, inspect the files under `data/processed/<prefix>_fold*/` if
you need to verify subject assignments.

## Aggregating CV Results

Training outputs are saved under the normal benchmark directory:

```text
runs/benchmark/<dataset_fold>/<mode>/<model>/model_seed_<seed>/metrics.json
```

For example:

```text
runs/benchmark/sisfall_20hz_fold0/classification/random_forest/model_seed_42/metrics.json
```

Aggregate the test metrics across folds with:

```bash
python -m src.training.aggregate_cv_metrics \
  --dataset-prefix sisfall_20hz \
  --n-folds 5 \
  --mode classification \
  --model random_forest \
  --model-seed 42
```

The aggregator reads only the `test` section from each fold's `metrics.json`.
Validation metrics are not final benchmark results.

Outputs are written to:

```text
runs/benchmark/sisfall_20hz_cv/classification/random_forest/model_seed_42/
```

Key files:

- `cv_metrics.json`: fold metrics plus mean/std summary
- `cv_metrics.csv`: compact mean/std table
- `fold_metrics.csv`: per-fold test metrics

## Reporting

Report mean and standard deviation across folds for the final table:

- AUROC
- AUPRC
- Precision
- Recall / Sensitivity
- Specificity
- F1

Use `n_folds=5` for the main benchmark across SisFall, FallAllD, UMAFall, and
UP-FALL. LOSO is not implemented in the current workflow and should not block
the main results.
