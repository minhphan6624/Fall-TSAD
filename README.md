# Fall-TSAD: Time-Series Anomaly Detection for Fall Events

## Description
This project implements a deep learning approach for time-series anomaly detection, specifically targeting the identification of fall events from sensor data. The project is built with PyTorch and uses Hydra for flexible configuration management.

## Project Structure
The project is organized into several key directories:
*   `src/`: Contains the core source code, including definitions for models (`src/models`), preprocessing logic (`src/preprocessing`), and training components (`src/trainers`).
*   `scripts/`: Houses shell scripts to streamline common operations such as data preprocessing, model training, and evaluation.
*   `notebooks/`: Includes Jupyter notebooks for exploratory data analysis (EDA) and other experimental work.
*   `docs/`: Provides detailed documentation, including a guide on data preprocessing.
*   `runs/`: Stores the results of experiments, including trained model checkpoints, evaluation reports, and training logs.
*   Top-level files: `.gitignore`, `README.md`, `requirements.txt`, `eval_lstm_ae.py` (for evaluation), and `train_lstm_ae.py` (for training).

## Installation
To set up the project environment and install the necessary dependencies, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/minhphan6624/Fall-TSAD.git
    cd Fall-TSAD
    ```

2.  **Create a virtual environment (recommended):**

    **Using `venv`:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

    **Using `conda`:**
    ```bash
    conda create -n fall-tsad python=3.9 # Or your preferred Python version
    conda activate fall-tsad
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project involves a multi-stage pipeline for time-series anomaly detection, encompassing data preprocessing, model training, and evaluation. The primary scripts for these stages are located in the `scripts/` directory.

### 1. Data Preprocessing
The preprocessing pipeline prepares raw sensor data for model training. This involves steps like loading metadata, filtering signals, per-sensor normalization, and segmenting data into windows with appropriate labels. More details can be found in `docs/data_preprocessing.md`.

To run the preprocessing pipeline:
```bash
./scripts/preprocess.sh
```
You can specify different data configurations using Hydra (e.g., `./scripts/preprocess.sh data=upfall`).

### 2. Model Training
The training script is used to train the LSTM Autoencoder model.

To train the model with default configurations:
```bash
./scripts/train.sh
```

### 3. Model Evaluation
After training, the evaluation script assesses the trained model's performance, often involving threshold selection and metric calculation at both window and event levels.

To evaluate the model:
```bash
./scripts/evaluate.sh
```
(Note: This script might require specific arguments or configurations, such as the path to the trained model and selected threshold. Refer to the script for details.)

## Configuration
This project uses [Hydra](https://hydra.cc/) for managing configurations. All configuration files are located in the `configs/` directory.

The `configs/default.yaml` file specifies the default settings for the dataset, model, and trainer. You can override these defaults or specify different configurations using command-line arguments or by creating new configuration files.

**Example: Overriding model parameters**
To change a model parameter (e.g., `latent_dim` for the `lstm_ae` model), you can run:
```bash
python train_lstm_ae.py model.latent_dim=16
```

**Example: Using a different dataset configuration**
If you have a `configs/data/upfall.yaml` configuration, you can use it by running:
```bash
python train_lstm_ae.py data=upfall
```

You can combine multiple overrides as needed.
