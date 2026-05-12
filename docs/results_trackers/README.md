# Results Tracker Schemas

Use these CSV files to track final CV results before moving them into the report.
Each row should come from one aggregated CV output directory:

```text
runs/benchmark/<prefix>_cv/<mode>/<model>/model_seed_<seed>/
```

Use `cv_metrics.csv` as the source for mean/std values. Use `fold_metrics.csv`
only when inspecting fold-level variation.

## Shared Columns

- `dataset`: `sisfall`, `fallalld`, `umafall`, or `upfall`
- `prefix`: dataset prefix used before `_fold0.._fold4`
- `model`: run-output model name, such as `random_forest` or `lstm_ae`
- `mode`: `classification` or `tsad`
- metric columns: mean/std copied from the aggregated `cv_metrics.csv`

The seed, fold count, window size, sampling rate, and ADL condition are inferred
from the project defaults and the `variant` / `prefix` values.

## Tracker Files

- `main_benchmark.csv`: primary 20 Hz, 2s benchmark across datasets/models
- `window_duration.csv`: compares 20 Hz 2s, 3s, and 5s windows
- `sampling_rate.csv`: compares native 2s against 20 Hz 2s
- `easy_adl.csv`: compares normal 20 Hz 2s against easy-ADL 20 Hz 2s

For report tables, format important metrics as `mean ± std`, for example
`0.842 ± 0.061`.
