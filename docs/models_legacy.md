# Legacy Model Selection Notes

This file preserves the previous `docs/models.md` content before `docs/models.md` was updated to track the current implemented setup.

# Model Selection Notes

This document summarizes the model choices for the fall-detection benchmark comparing time-series anomaly detection (TSAD) methods against supervised classification methods on triaxial accelerometer windows.

The primary benchmark uses all datasets downsampled to 20 Hz. The secondary benchmark uses each dataset at its native sampling rate and is reported separately as a sensitivity analysis.

## Benchmark Assumptions

The primary benchmark fixes the input resolution before model selection:

- sampling rate: 20 Hz
- window duration: 2 seconds
- channels: 3 accelerometer axes
- raw-window input shape: `40 x 3`
- flattened raw-window input size: `120`

Window duration should be defined in seconds rather than samples. This keeps the physical time span consistent when comparing downsampled and native-rate experiments.

The same subject-disjoint splits, windowing policy, labeling rule, and evaluation protocol should be used for classification and TSAD. The learning setup differs:

- classification models train on fall and non-fall windows
- TSAD models train on normal windows only

## Methodology Model Summary

For the methodology section, the model set can be summarized by learning setup, input representation, and scoring output rather than implementation status:

| Model | Model family | Learning setup | Training windows | Input representation | Output score |
|---|---|---|---|---|---|
| Signal-threshold baseline | Rule-based baseline | Heuristic | No fitted training stage | Acceleration magnitude or engineered signal features | Thresholded fall decision |
| Random Forest | Classical supervised classifier | Binary classification | Fall and non-fall windows | Engineered window features | Fall probability or class score |
| XGBoost or SVM | Classical supervised classifier | Binary classification | Fall and non-fall windows | Engineered window features | Fall probability or class score |
| 1D CNN classifier | Deep supervised classifier | Binary classification | Fall and non-fall windows | Normalized raw `40 x 3` accelerometer windows | Sigmoid fall score |
| LSTM classifier | Deep supervised classifier | Binary classification | Fall and non-fall windows | Normalized raw `40 x 3` accelerometer windows | Sigmoid fall score |
| Isolation Forest | Classical TSAD | Normal-only anomaly detection | Normal windows only | Engineered window features | Anomaly score |
| Dense autoencoder | Neural TSAD | Normal-only reconstruction | Normal windows only | Flattened raw window, `120` values | Reconstruction error |
| LSTM autoencoder | Neural TSAD | Normal-only reconstruction | Normal windows only | Normalized raw `40 x 3` accelerometer windows | Reconstruction error |
| Conv1D autoencoder | Neural TSAD | Normal-only reconstruction | Normal windows only | Normalized raw `40 x 3` accelerometer windows | Reconstruction error |

The primary comparison should focus on the supervised classifiers and the core TSAD models. The signal-threshold baseline, Conv1D autoencoder, VAE, LSTM-VAE, and TranAD can be described as secondary or optional comparisons if they are included after the main benchmark pipeline is stable.

## Current Implementation Status

The current model architecture files are under:

```text
src/models/
```

Implemented neural architecture files:

```text
cnn1d.py
cnn1d_large.py
cnn1d_ae.py
cnn1d_ae_large.py
lstm_classifier.py
dense_ae.py
lstm_ae.py
```

Current deep training entry points are under:

```text
src/training/train_cnn1d.py
src/training/train_cnn1d_ae.py
src/training/train_cnn1d_ae_large.py
src/training/train_lstm_classifier.py
src/training/train_dense_ae.py
src/training/train_lstm_ae.py
```

Shared PyTorch training, scoring, model-selection, and checkpoint helpers are in:

```text
src/training/deep_utils.py
```

The deep classifier scripts train with `BCEWithLogitsLoss` and output sigmoid fall scores. The deep autoencoder scripts train with MSE reconstruction loss on TSAD normal-only training windows and score validation/test windows by mean squared reconstruction error.

## Core Classification Models

The primary classification benchmark should include:

- feature-threshold or signal-threshold baseline
- Random Forest on engineered features
- XGBoost/SVM on the same engineered features
- 1D CNN classifier on raw normalized windows
- LSTM classifier on raw normalized windows

The feature-threshold baseline can be added after the main model pipeline is stable. It is useful because fall detection has strong acceleration-magnitude heuristics, and the learned models should be compared against a simple impact-based rule.

### Engineered Features for feature-based models

Random Forest, XGBoost, and Isolation Forest should use the same engineered feature table so the comparison reflects the learning algorithm rather than feature availability.

The current feature definitions are documented in `docs/features.md` and implemented in `src/training/extract_features.py`.

### Random Forest

Purpose: robust classical supervised baseline.

Input:

```text
n_windows x n_features
```

Recommended default:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=seed,
    n_jobs=-1,
)
```

Small tuning grid:

```text
n_estimators: [300]
max_depth: [5, 10, 20, None]
min_samples_leaf: [1, 2, 5]
max_features: ["sqrt", 0.5]
```

### XGBoost

Purpose: stronger engineered-feature classifier. XGBoost is worth including even with a single triaxial sensor because the model operates on the engineered window-feature table, not directly on the number of sensors.

However, can consider switching to svm

It is only an extension anyways

Input:

```text
n_windows x n_features
```

Recommended default:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=normal_count / fall_count,
    random_state=seed,
)
```

Small tuning grid:

```text
max_depth: [3, 4, 5]
learning_rate: [0.03, 0.05, 0.1]
n_estimators: [100, 300, 500]
subsample: [0.8, 1.0]
colsample_bytree: [0.8, 1.0]
```

Use validation-set early stopping if the implementation supports it cleanly. Because fall datasets are usually small, XGBoost should only be tuned on the subject-disjoint validation split.

### 1D CNN Classifier

Purpose: discriminative raw-window temporal baseline.

Input:

```text
40 x 3
```

Recommended default:

```text
Conv1D(32, kernel_size=5, padding="same")
BatchNorm
ReLU
MaxPool1D(pool_size=2)

Conv1D(64, kernel_size=3, padding="same")
BatchNorm
ReLU
GlobalAveragePooling1D

Dense(64)
Dropout(0.3)
Dense(1, activation="sigmoid")
```

Rationale:

- at 20 Hz, kernel size 5 spans 0.25 seconds
- at 20 Hz, kernel size 3 spans 0.15 seconds
- global average pooling avoids hard-coding a flattened temporal size
- the model is small enough for limited fall-detection datasets

Small tuning grid:

```text
filters: [(16, 32), (32, 64), (64, 128)]
first kernel size: [3, 5, 7]
dropout: [0.2, 0.3, 0.5]
dense units: [32, 64]
learning_rate: [1e-3, 3e-4]
```

### LSTM Classifier

Purpose: recurrent raw-window sequence baseline.

Default arch:

```text
LSTM(64, return_sequences=False)
Dropout(0.3)
Dense(32, activation="relu")
Dropout(0.2)
Dense(1, activation="sigmoid")
```

Smaller fallback:

```text
LSTM(32, return_sequences=False)
Dropout(0.3)
Dense(1, activation="sigmoid")
```

Small tuning grid:

```text
hidden units: [32, 64]
dropout: [0.2, 0.3, 0.5]
dense units: [0, 32]
learning_rate: [1e-3, 3e-4]
```

If single-layer LSTM underfits, used stacked LSTM, otherwise don't

## Core TSAD Models

Primary:

- Isolation Forest on engineered features
- LSTM autoencoder on raw windows
- Dense autoencoder on flattened raw windows

AFter the main TSAD are working, add Conv-AE and TranAD 

For every TSAD model:

- train only on normal training windows
- compute continuous anomaly scores for validation and test windows
- report AUROC and AUPRC from continuous scores
- choose operating thresholds using validation data only
- report thresholded Precision, Recall, F1, Sensitivity, and Specificity on the test split

### Isolation Forest

Purpose: non-deep TSAD baseline.

Input: engineered features (decided later)

Use the same engineered feature family as RF and XGBoost. The anomaly score is the negative Isolation Forest decision score.

Default setup:

```python
IsolationForest(
    n_estimators=300,
    contamination="auto",
    max_samples="auto",
    max_features=1.0,
    random_state=seed,
    n_jobs=-1,
)
```

Small tuning grid:

```text
n_estimators: [200, 300, 500]
max_samples: ["auto", 0.5, 0.8]
```

The operating threshold should be selected on the validation split rather than assumed from the contamination parameter.

### Dense Autoencoder

Purpose: simple neural reconstruction baseline.

Input: raw window 

Recommended default:

```text
Input(120)
Dense(128), ReLU
Dropout(0.1)
Dense(64), ReLU
Dense(16), ReLU
Dense(64), ReLU
Dense(128), ReLU
Dense(120), linear output
```

Loss: MSE reconstruction loss

Anomaly score: mean squared reconstruction error over time and axes

Tuning grid:

```text
latent_dim: [8, 16, 32]
hidden sizes: [[64], [128, 64]]
dropout: [0.0, 0.1, 0.2]
learning_rate: [1e-3, 3e-4]
```

The Dense AE ignores temporal order after flattening, so it should be treated as a simple baseline rather than the expected best neural TSAD model.

### Conv1D Autoencoder

Purpose: convolutional reconstruction baseline aligned with the 1D CNN classifier.

Input: Raw 40x3 accelerometer data

Recommended default:

```text
Conv1D(32, kernel_size=5, padding="same")
ReLU
MaxPool1D(pool_size=2)       # 40 -> 20

Conv1D(64, kernel_size=3, padding="same")
ReLU
MaxPool1D(pool_size=2)       # 20 -> 10

Conv1D(64, kernel_size=3, padding="same")
ReLU

UpSampling1D(size=2)         # 10 -> 20
Conv1D(32, kernel_size=3, padding="same")
ReLU

UpSampling1D(size=2)         # 20 -> 40
Conv1D(3, kernel_size=3, padding="same")
Linear output
```

Loss:  MSE reconstruction loss

Anomaly score: mean squared reconstruction error over time and axes

This model is often a useful middle ground between Dense AE and LSTM-AE because it captures local temporal patterns while remaining small and fast.

### LSTM Autoencoder

Purpose: sequence reconstruction baseline that preserves temporal ordering.

Input: 40 x 3 accelerometer data

Default:

```text
LSTM(64, return_sequences=False)
Dropout(0.2)
Dense(32), ReLU
RepeatVector(40)
LSTM(64, return_sequences=True)
TimeDistributed(Dense(3))
```

Smaller fallback:

```text
LSTM(32, return_sequences=False)
Dense(16), ReLU
RepeatVector(40)
LSTM(32, return_sequences=True)
TimeDistributed(Dense(3))
```

Loss: MSE reconstruction loss

Anomaly score: mean squared reconstruction error over time and axes

Grid:

```text
lstm_units: [32, 64]
latent_dim: [16, 32]
dropout: [0.0, 0.2, 0.3]
learning_rate: [1e-3, 3e-4]
```

## Optional TSAD Models

### VAE

Input:

```text
flattened 40 x 3 window = 120
```

Reference architecture:

```text
Input(120)
Dense(128), ReLU
Dense(64), ReLU
z_mean(16), z_log_var(16)
Sampling
Dense(64), ReLU
Dense(128), ReLU
Dense(120), linear output
```

Use reconstruction error as the first anomaly score for consistency with the other autoencoders. Negative ELBO can be evaluated later if probabilistic scoring becomes part of the study.

### LSTM-VAE

Reference architecture:

```text
LSTM(64, return_sequences=False)
z_mean(16), z_log_var(16)
Sampling
RepeatVector(40)
LSTM(64, return_sequences=True)
TimeDistributed(Dense(3))
```

This model is optional because it adds tuning complexity and may behave inconsistently on small fall datasets.

### Transformer-Based TSAD

Transformer-based TSAD models are not excluded, but they are treated as advanced comparison models rather than first-wave baselines.

1. TranAD

Reasoning:

- the fall datasets are relatively small and low-dimensional compared with many industrial multivariate TSAD benchmarks
- the primary 20 Hz setting produces short sequences such as `40 x 3`
- transformer TSAD models add tuning sensitivity
- they are harder to interpret fairly before the benchmark protocol is stable

Use default published architectures first and tune minimally.

## Secondary Native-Rate Benchmark

The secondary benchmark keeps each dataset at its native sampling rate and uses the same window duration in seconds. This means the raw input length changes by dataset:

```text
20 Hz, 2 s    -> 40 x 3
200 Hz, 2 s   -> 400 x 3
238 Hz, 2 s   -> 476 x 3
```

This benchmark should be reported separately because native sampling rate can affect model performance. A model may improve because it receives higher-resolution temporal detail, not because it is generally better.

Architecture guidance for the secondary benchmark:

- CNN and Conv1D-AE should use global pooling or sequence-length-tolerant layers where possible
- LSTM and LSTM-AE can support variable sequence lengths but may require padding or dataset-specific batching
- Dense AE and VAE are less convenient because flattened input size changes with sampling rate
- RF, XGBoost, and Isolation Forest should use duration-based, rate-aware engineered features

The primary 20 Hz benchmark remains the main fair model-family comparison.
