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