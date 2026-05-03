# Subject-Wise K-Fold and LOSO Evaluation Options

This document records practical implementation options for adding stronger
subject-wise evaluation to the project.

The current development pipeline uses one subject-disjoint train/validation/test
split. That is useful for development, but final benchmark results should use a
more robust subject-wise protocol, preferably 5-fold cross-validation.

## Evaluation Goal

The benchmark should test whether models generalize to unseen subjects.

The required rule is:

- no subject can appear in more than one of train, validation, or test
- model fitting uses the training split only
- threshold selection and hyperparameter selection use the validation split only
- final reported metrics use the test split only

This applies to both supervised classification and TSAD models.

## Recommended Final Protocol

Use subject-wise 5-fold cross-validation as the main final protocol.

For each fold:

- use one subject fold as test
- use one different subject fold as validation
- use all remaining subject folds as training

Then report the mean and standard deviation of test metrics across folds.

Recommended final table metrics:

- AUROC
- AUPRC
- Precision
- Recall / Sensitivity
- Specificity
- F1

LOSO can be added as a secondary robustness analysis if compute time allows.
It is methodologically strong, but can be noisy because each test set contains
only one held-out subject.

## Option 1: Manual Split CSVs

This is already supported by the current preprocessing entry point.

Relevant script:

```text
src/preprocessing/run_pipeline_2.py
```

Relevant argument:

```bash
--manual-split-csv
```

Each CSV needs at least:

```text
subject_id,split
SA01,train
SA02,val
SA03,test
```

Example 5-fold preprocessing command:

```bash
python -m src.preprocessing.run_pipeline_2 \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --manual-split-csv splits/sisfall_fold0.csv \
  --output-dataset sisfall_20hz_fold0
```

Then train normally against the generated fold dataset:

```bash
python -m src.training.train_lstm_ae --dataset sisfall_20hz_fold0 --model-seed 42 --device auto
```

Advantages:

- minimal code changes
- easy to inspect exact subject assignments
- useful for debugging and reproducibility

Disadvantages:

- easy to make mistakes when hand-writing many CSVs
- creates extra split files to maintain
- tedious for LOSO

Use this if the immediate goal is to run final results quickly with minimal new
implementation.

## Option 2: Automatic Fold Generation in Preprocessing

This is the cleanest next code change.

Add split-protocol arguments to `run_pipeline_2.py`, for example:

```bash
--split-protocol subject_kfold
--n-folds 5
--fold-index 0
```

or:

```bash
--split-protocol loso
--test-subject SA01
```

Example target command:

```bash
python -m src.preprocessing.run_pipeline_2 \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --split-protocol subject_kfold \
  --n-folds 5 \
  --fold-index 0 \
  --output-dataset sisfall_20hz_fold0
```

Example LOSO target command:

```bash
python -m src.preprocessing.run_pipeline_2 \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --split-protocol loso \
  --test-subject SA01 \
  --output-dataset sisfall_20hz_loso_SA01
```

Suggested implementation:

- add `build_subject_kfold_splits(...)` to `src/preprocessing/build_splits.py`
- add `build_loso_split(...)` to `src/preprocessing/build_splits.py`
- keep the current default percentage split for development
- keep `--manual-split-csv` as an override

For k-fold assignment, use subject-level rows from `subject_summary.csv` style
data:

```text
subject_id
has_fall
n_trials
n_fall_trials
n_adl_trials
```

Prefer stratifying by `has_fall` when possible so each fold has fall-capable
subjects. If the dataset is too small for strict stratification, fall-subject
coverage should be prioritized manually in code.

Advantages:

- avoids hand-written split CSVs
- keeps the existing processed-artifact training workflow
- makes every fold inspectable under `data/processed/`
- small code change relative to a full training refactor

Disadvantages:

- still requires running preprocessing once per fold
- still creates one processed folder per fold

This is the recommended implementation path for this project.

## Option 3: Dedicated CV Preprocessing Runner

Instead of passing `--fold-index` manually, add a script such as:

```text
src/preprocessing/run_cv_pipeline.py
```

The script would:

1. load the interim trial dataframe
2. build all fold split definitions
3. call `run_pipeline(...)` from `run_pipeline_2.py` once per fold
4. save outputs such as:

```text
data/processed/sisfall_20hz_fold0/
data/processed/sisfall_20hz_fold1/
data/processed/sisfall_20hz_fold2/
data/processed/sisfall_20hz_fold3/
data/processed/sisfall_20hz_fold4/
```

Example target command:

```bash
python -m src.preprocessing.run_cv_pipeline \
  --dataset sisfall \
  --target-sampling-rate-hz 20 \
  --protocol subject_kfold \
  --n-folds 5 \
  --output-prefix sisfall_20hz
```

Advantages:

- one command generates all folds
- less repetitive than running `run_pipeline_2.py` per fold
- training scripts remain unchanged

Disadvantages:

- slightly more implementation work than Option 2
- still creates processed artifacts per fold

This is a good follow-up after Option 2 works.

## Option 4: Full Training CV Runner

Add a higher-level training script such as:

```text
src/training/run_cv_benchmark.py
```

The script would:

1. generate or load fold definitions
2. preprocess each fold
3. train selected models
4. save fold-specific outputs
5. aggregate final metrics

Example target command:

```bash
python -m src.training.run_cv_benchmark \
  --dataset sisfall \
  --protocol subject_kfold \
  --n-folds 5 \
  --models random_forest isolation_forest cnn1d lstm_ae \
  --target-sampling-rate-hz 20
```

Advantages:

- most convenient once stable
- can generate final benchmark tables automatically
- reduces repeated shell commands

Disadvantages:

- largest refactor
- couples preprocessing and training
- harder to debug at first

Do this later, after the fold artifact workflow is stable.

## K-Fold Details

For 5-fold subject-wise validation:

- build folds from unique `subject_id` values
- assign complete subjects, not windows, to folds
- use fold `i` as test
- use fold `(i + 1) % n_folds` as validation
- use the remaining folds as training

Example for fold 0:

```text
fold 0 subjects -> test
fold 1 subjects -> val
fold 2, 3, 4 subjects -> train
```

Example for fold 1:

```text
fold 1 subjects -> test
fold 2 subjects -> val
fold 0, 3, 4 subjects -> train
```

This ensures every subject appears in the test split exactly once across the
full 5-fold run.

## LOSO Details

For leave-one-subject-out:

- choose one subject as test
- choose one or more different subjects as validation
- use all remaining subjects as training

Example:

```text
SA01 -> test
SA02 -> val
all other subjects -> train
```

The validation subject can rotate deterministically, for example the next
subject in sorted order. If fall labels are sparse, prefer choosing a validation
subject that contains fall trials when possible.

LOSO produces many runs:

```text
n_datasets x n_subjects x n_models x n_model_seeds
```

For that reason, it is best treated as a secondary analysis unless compute time
is not a concern.

## Aggregating Results

Each training script saves:

```text
runs/benchmark/<dataset>/<mode>/<model>/model_seed_<seed>/metrics.json
```

For fold datasets, the dataset name should include the fold:

```text
runs/benchmark/sisfall_20hz_fold0/tsad/lstm_ae/model_seed_42/metrics.json
```

Aggregate the `test` section from each `metrics.json`.

Report:

```text
mean metric across folds
standard deviation across folds
number of folds
```

Do not use validation metrics as final benchmark results. Validation metrics are
for threshold selection and hyperparameter selection only.

## Recommended Implementation Order

1. Add automatic k-fold and LOSO split builders in `build_splits.py`.
2. Add split-protocol arguments to `run_pipeline_2.py`.
3. Generate one fold and verify its subject assignments.
4. Train one shallow and one deep model on that fold.
5. Generate all 5 folds for one dataset.
6. Train the full model set on one dataset.
7. Add a small metrics aggregation script.
8. Repeat across all datasets.

Keep the existing single split workflow as the quick development path.
