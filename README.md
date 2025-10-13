# Fall-TSAD: Time-Series Anomaly Detection for Fall Events

## Description
This project implements a deep learning approach for time-series anomaly detection, specifically targeting the identification of fall events from sensor data.  The project is built with PyTorch and uses Hydra for flexible configuration management.

## Project Structure
```
.
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── data/
│   │   ├── sisfall.yaml
│   │   └── upfall.yaml
│   ├── model/
│   │   └── lstm_ae.yaml
│   └── trainer/
│       └── base.yaml
├── notebooks/
│   └── data_preprocessing.ipynb
├── scripts/
│   ├── preprocess.py
│   └── train.py
└── src/
    ├── datasets/
    │   ├── data_loader.py
    │   └── sisfall_dataset.py
    ├── modeling/
    │   ├── evaluate.py
    │   ├── select_threshold.py
    │   └── trainer.py
    ├── models/
    │   └── lstm_ae.py
    ├── pipelines/
    │   └── sisfall/
    │       ├── aa_run_pipeline.py
    │       ├── build_metadata.py
    │       ├── load_signal.py
    │       ├── normalize.py
    │       ├── segment_data.py
    │       ├── serialize.py
    │       └── split.py
    ├── utils/
    │   ├── registry.py
    │   └── set_seed.py
```

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

This project involves several stages: data preprocessing, model training, threshold selection, and model evaluation.

### 1. Data Preprocessing
The `preprocess.py` script is used to prepare the raw sensor data for model training.

To run the preprocessing pipeline:
```bash
python scripts/preprocess.py
```
You can specify different data configurations using Hydra (e.g., `python scripts/preprocess.py data=upfall`).

### 2. Model Training
The `train.py` script is used to train the LSTM Autoencoder model.

To train the model with default configurations:
```bash
python scripts/train.py
```

### 3. Threshold Selection
After training, the `select_threshold.py` script helps in determining an optimal anomaly threshold.

To select the threshold:
```bash
python src/modeling/select_threshold.py
```
(Note: This script might require specific arguments or configurations depending on its implementation. Refer to the script for details.)

### 4. Model Evaluation
The `evaluate.py` script is used to evaluate the trained model's performance.

To evaluate the model:
```bash
python src/modeling/evaluate.py
```
(Note: This script might require specific arguments or configurations, such as the path to the trained model and selected threshold. Refer to the script for details.)

## Configuration
This project uses [Hydra](https://hydra.cc/) for managing configurations. All configuration files are located in the `configs/` directory.

The `configs/default.yaml` file specifies the default settings for the dataset, model, and trainer. You can override these defaults or specify different configurations using command-line arguments or by creating new configuration files.

**Example: Overriding model parameters**
To change a model parameter (e.g., `latent_dim` for the `lstm_ae` model), you can run:
```bash
python scripts/train.py model.latent_dim=16
```

**Example: Using a different dataset configuration**
If you have a `configs/data/upfall.yaml` configuration, you can use it by running:
```bash
python scripts/train.py data=upfall
```

You can combine multiple overrides as needed.
