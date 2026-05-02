# Results Summary

This document records curated benchmark/checkpoint results copied from local run artifacts under `runs/`.

Generated run artifacts under `runs/` are not version controlled. Keep the run folders local and copy key results here when they are useful for later reference.

## Current Benchmark Context

- Run root: `runs/benchmark`
- Datasets: `fallalld_20hz`, `sisfall_20hz`, `umafall_20hz`, `upfall_20hz`
- Sampling setup: all listed benchmark datasets are downsampled to 20 Hz.
- Windowing: 2 second windows with 50% overlap, giving 40 samples per window.
- Split protocol: subject-disjoint train/validation/test split, split seed 7, target ratio 70/15/15.
- Threshold policy: operating threshold is selected on the validation split by best F1, then reused on test.
- Metrics below are window-level test metrics.
- Shallow models use engineered features from normalized windows. CNN1D uses raw normalized `40 x 3` windows.
- Classification normalization is fit on all training windows. TSAD normalization is fit on normal training windows only, and TSAD models train on normal training windows only.

## Processed Split Sizes

Values are `total / normal / fall` window counts for the classification processed split.

| Dataset | Subjects train/val/test | Train windows | Val windows | Test windows |
| --- | ---: | ---: | ---: | ---: |
| `fallalld_20hz` | 10 / 2 / 2 | 25061 / 24725 / 336 | 4313 / 4187 / 126 | 4788 / 4728 / 60 |
| `sisfall_20hz` | 26 / 6 / 6 | 45269 / 43882 / 1387 | 14676 / 13963 / 713 | 14720 / 14029 / 691 |
| `umafall_20hz` | 12 / 3 / 3 | 5197 / 5042 / 155 | 1377 / 1322 / 55 | 1736 / 1659 / 77 |
| `upfall_20hz` | 11 / 3 / 3 | 9915 / 9658 / 257 | 2815 / 2742 / 73 | 2811 / 2741 / 70 |

## Benchmark Test Results

| Dataset | Mode | Model | Seed | AUROC | AUPRC | Precision | Recall | Specificity | F1 | Val-selected threshold | Test confusion matrix |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fallalld_20hz` | classification | Random Forest | 42 | 0.998 | 0.944 | 0.945 | 0.867 | 0.999 | 0.904 | 0.310 | TN=4725, FP=3, FN=8, TP=52 |
| `fallalld_20hz` | classification | XGBoost | 42 | 0.999 | 0.953 | 0.926 | 0.833 | 0.999 | 0.877 | 0.966 | TN=4724, FP=4, FN=10, TP=50 |
| `fallalld_20hz` | TSAD | Isolation Forest | 42 | 0.984 | 0.346 | 0.322 | 0.817 | 0.978 | 0.462 | 0.110 | TN=4625, FP=103, FN=11, TP=49 |
| `sisfall_20hz` | classification | CNN1D | 42 | 0.999 | 0.963 | 0.874 | 0.974 | 0.993 | 0.921 | 0.963 | TN=13932, FP=97, FN=18, TP=673 |
| `sisfall_20hz` | classification | Random Forest | 42 | 1.000 | 0.996 | 0.987 | 0.964 | 0.999 | 0.975 | 0.463 | TN=14020, FP=9, FN=25, TP=666 |
| `sisfall_20hz` | classification | XGBoost | 42 | 1.000 | 0.996 | 0.985 | 0.952 | 0.999 | 0.968 | 0.966 | TN=14019, FP=10, FN=33, TP=658 |
| `sisfall_20hz` | TSAD | Isolation Forest | 42 | 0.985 | 0.657 | 0.637 | 0.790 | 0.978 | 0.705 | 0.141 | TN=13718, FP=311, FN=145, TP=546 |
| `umafall_20hz` | classification | Random Forest | 42 | 0.998 | 0.960 | 0.861 | 0.883 | 0.993 | 0.872 | 0.377 | TN=1648, FP=11, FN=9, TP=68 |
| `umafall_20hz` | classification | XGBoost | 42 | 0.999 | 0.972 | 0.930 | 0.857 | 0.997 | 0.892 | 0.866 | TN=1654, FP=5, FN=11, TP=66 |
| `umafall_20hz` | TSAD | Isolation Forest | 42 | 0.927 | 0.249 | 0.264 | 0.922 | 0.881 | 0.410 | 0.100 | TN=1461, FP=198, FN=6, TP=71 |
| `upfall_20hz` | classification | Random Forest | 42 | 0.999 | 0.946 | 0.906 | 0.829 | 0.998 | 0.866 | 0.500 | TN=2735, FP=6, FN=12, TP=58 |
| `upfall_20hz` | classification | XGBoost | 42 | 0.999 | 0.950 | 0.880 | 0.943 | 0.997 | 0.910 | 0.884 | TN=2732, FP=9, FN=4, TP=66 |
| `upfall_20hz` | TSAD | Isolation Forest | 42 | 0.933 | 0.153 | 0.180 | 0.829 | 0.903 | 0.295 | 0.091 | TN=2476, FP=265, FN=12, TP=58 |

## Quick Takeaways

- Best test F1 by dataset:
  - `fallalld_20hz`: Random Forest, F1 0.904.
  - `sisfall_20hz`: Random Forest, F1 0.975.
  - `umafall_20hz`: XGBoost, F1 0.892.
  - `upfall_20hz`: XGBoost, F1 0.910.
- Supervised classification models currently outperform Isolation Forest TSAD on F1 for every dataset.
- Isolation Forest often keeps high recall, especially on `umafall_20hz`, but with many more false positives and much lower precision.
- SisFall has the strongest overall benchmark results among the current completed runs.

## Smoke Run

The following run is kept separate from benchmark results because it was a development smoke run.

| Run root | Dataset | Mode | Model | Seed | Training setup | Test AUROC | Test AUPRC | Test Precision | Test Recall | Test Specificity | Test F1 | Notes |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/benchmark_smoke` | `umafall_20hz` | classification | CNN1D | 42 | 1 epoch, batch size 256 | 0.880 | 0.150 | 0.200 | 0.805 | 0.851 | 0.320 | Pipeline/checkpoint sanity check only; not a final benchmark result. |

## What To Record Next

For future selected results, record:

- dataset and sampling setup
- split protocol and split seed
- learning mode and model name
- model seed and important hyperparameters
- threshold policy
- AUROC, AUPRC, precision, recall/sensitivity, specificity, F1
- confusion matrix
- important caveats, especially smoke runs, incomplete training, or non-comparable preprocessing

Do not paste full prediction tables, model checkpoints, TensorBoard logs, or generated plots into this document.
