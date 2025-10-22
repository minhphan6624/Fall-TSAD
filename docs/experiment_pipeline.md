# Fall Detection Anomaly Detection Pipeline Summary

This document summarizes the end-to-end pipeline for fall detection using anomaly detection on the SisFall dataset, covering data processing, model implementation, threshold selection, and evaluation.

## 1. Data Processing for SisFall

The data processing pipeline for the SisFall dataset involves several steps orchestrated by `scripts/preprocess.py` and defined in `src/pipelines/sisfall/`. The configuration for data processing is specified in `configs/data/sisfall.yaml`.

**Configuration (`configs/data/sisfall.yaml`):**
- `name`: `sisfall`
- `raw_dir`: `data/raw/sisfall/`
- `processed_dir`: `data/processed/sisfall/`
- `batch_size`: `32`
- `sampling_rate`: `200` Hz
- `segment_seconds`: `3` (segments are 3 seconds long)
- `overlap`: `0.5` (50% overlap between segments)
- `normalize`: `zscore` (data is normalized using z-score standardization)
- `split`: `train: 0.7`, `val: 0.15`, `test: 0.15` (data is split into training, validation, and test sets)
- `random_seed`: `42`

**Pipeline Steps (`src/pipelines/sisfall/`):**
- `aa_run_pipeline.py`: Main entry point for the SisFall data processing pipeline.
- `build_metadata.py`: Likely responsible for creating metadata files for the dataset.
- `load_signal.py`: Handles loading raw sensor signals.
- `normalize.py`: Implements the z-score normalization as specified in the config.
- `segment_data.py`: Segments the continuous time series data into fixed-length windows with overlap.
- `serialize.py`: Saves the processed data into a format suitable for model training (e.g., `.npy` files).
- `split.py`: Divides the dataset into training, validation, and test sets based on the configured ratios.

**Dataset Loading (`src/datasets/sisfall_dataset.py`):**
The `SisFallDataset` class loads the preprocessed `.npy` files for each split (train, val, test) and provides them as `torch.Tensor` objects. Each item in the dataset consists of a data segment (`X`) and its corresponding label (`y`).

## 2. Model Implementation (LSTM Autoencoder)

The model used for anomaly detection is an LSTM Autoencoder, configured via `configs/model/lstm_ae.yaml` and implemented in `src/models/lstm_ae.py`.

**Configuration (`configs/model/lstm_ae.yaml`):**
- `n_features`: `6` (number of input features, e.g., accelerometer x, y, z and gyroscope x, y, z)
- `hidden_dim`: `64` (dimension of the hidden state in LSTM layers)
- `num_layers`: `2` (number of recurrent layers)
- `dropout`: `0.2` (dropout rate for regularization)

**Model Architecture (`src/models/lstm_ae.py`):**
The `LSTM_AE` class consists of:
- An **Encoder LSTM**: Processes the input sequence (`n_features`) into a hidden representation (`hidden_dim`).
- A **Decoder LSTM**: Takes the last hidden state of the encoder, repeats it for the sequence length, and reconstructs the input sequence.
- An **Output Layer**: A linear layer that projects the decoder's output back to the original `n_features` dimension.
The model aims to reconstruct normal (non-fall) sequences accurately. Anomalies (falls) are expected to have high reconstruction errors.

## 3. Model Training

The model training is orchestrated by `scripts/train.py` and uses the `Trainer1` class from `src/modeling/trainer.py`.

**Training Process (`scripts/train.py` and `src/modeling/trainer.py`):**
- **Initialization**: The `Trainer1` initializes the LSTM-AE model, moves it to the appropriate device (MPS, CUDA, or CPU), sets up the `MSELoss` criterion, and an `Adam` optimizer with a learning rate defined in `cfg.trainer.learning_rate`.
- **Data Loaders**: `get_dataloaders(cfg)` creates `DataLoader` instances for the training and validation sets.
- **Training Loop (`fit` method)**:
    - Iterates for a specified number of `epochs` (`cfg.trainer.epochs`).
    - For each epoch, it calls `_train_epoch` and `_validate_epoch`.
    - **`_train_epoch`**: Sets the model to training mode, iterates through the `train_loader`, performs a forward pass, calculates MSE loss between input and reconstructed output, backpropagates the loss, and updates model weights using the optimizer.
    - **`_validate_epoch`**: Sets the model to evaluation mode (`torch.no_grad()`), iterates through the `val_loader`, calculates MSE loss, and returns the average validation loss.
    - **Checkpointing**: The model's state dictionary is saved as `last.pt` after each epoch. If the validation loss improves, the model is saved as `best_model.pt`.
    - **Logging**: Training and validation losses, along with epoch time, are logged and saved to `metrics.json`.

## 4. Threshold Selection

After training, an optimal anomaly threshold is selected using the validation set. This process is handled by `src/modeling/select_threshold.py`.

**Threshold Selection Logic (`src/modeling/select_threshold.py`):**
- **`calculate_reconstruction_errors`**:
    - Loads the best trained model (`best_model.pt`) from the experiment directory.
    - Iterates through the validation `DataLoader`, calculates the Mean Squared Error (MSE) between the original input and the model's reconstruction for each sequence.
    - Stores these reconstruction errors and their corresponding true labels.
- **`find_optimal_threshold`**:
    - Takes the calculated `errors` and `labels` from the validation set.
    - Iterates through all unique reconstruction error values as potential thresholds.
    - For each threshold, it classifies sequences as anomalous if their error is above the threshold.
    - Calculates the F1-score for each threshold.
    - The threshold that yields the highest F1-score is selected as the `optimal_threshold`. This approach prioritizes a balance between precision and recall.
- **Saving Threshold**: The `optimal_threshold` and its corresponding F1-score are saved to `optimal_threshold.json` within the experiment directory.

## 5. Evaluation

The final evaluation of the model's performance on unseen data (the test set) is performed by `src/modeling/evaluate.py`.

**Evaluation Process (`src/modeling/evaluate.py`):**
- **Loading Components**:
    - Loads the training configuration (`config.yaml`) and the best trained model (`best_model.pt`) from the experiment directory.
    - Loads the `optimal_threshold` from `optimal_threshold.json`.
- **Test Data**: Obtains the test `DataLoader` using `get_dataloaders(train_cfg)`.
- **Reconstruction Errors**: Calls `calculate_reconstruction_errors` on the test set to get `test_errors` and `test_labels`.
- **Prediction**: Applies the `optimal_threshold` to the `test_errors` to generate binary predictions (anomaly/normal).
- **Metric Calculation**: Calculates and logs the following metrics:
    - F1-Score
    - Precision
    - Recall
    - Confusion Matrix
- **Saving Results**: All evaluation results, including the metrics and confusion matrix, are saved to `evaluation_results.json` in the experiment directory.

| Feature | split.py | split1.py |
| :--- | :--- | :--- |
| Usage | Library function | Standalone script |
| Validation Set | Contains only ADL data from young adults. | Contains a mix of ADL and Fall data from young adults. |
| Test Set | Explicitly defined (all falls + elderly ADL). | "Everything else" not in train or validation sets. |
| Configuration | Via function arguments (val_size, seed). | Via global constants at the top of the file. |
| Features | Basic split. | Advanced features like subject-level holdouts. |
| Output | Returns a dictionary of DataFrames. | Writes DataFrames to CSV files and prints a report. |