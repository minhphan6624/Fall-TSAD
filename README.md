# Fall-TSAD: Time-Series Anomaly Detection for Fall Events

## Description
This project implements time-series anomaly detection and classification workflows for wearable fall-detection data, with a shared preprocessing pipeline and PyTorch-based models.

## Project Structure
The project is organized into several key directories:
*   `src/`: Contains the core source code, including definitions for models (`src/models`), preprocessing logic (`src/preprocessing`), and training components (`src/trainers`).
*   `scripts/`: Houses shell scripts to streamline common operations such as data preprocessing, model training, and evaluation.
*   `notebooks/`: Includes Jupyter notebooks for exploratory data analysis (EDA) and other experimental work.
*   `docs/`: Provides detailed documentation, including a guide on data preprocessing.
*   `runs/`: Stores the results of experiments, including trained model checkpoints, evaluation reports, and training logs.
*   Top-level files: `.gitignore`, `README.md`, `requirements.txt`

## Installation
To set up the project environment and install the necessary dependencies, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/minhphan6624/Fall-TSAD.git
    cd Fall-TSAD
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    On Windows:
    ```bash
    py -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Optional: install a CUDA-enabled PyTorch build**

    The default `requirements.txt` setup is intended to be simple and portable. If you want GPU acceleration, install the appropriate PyTorch build for your system from the official PyTorch instructions, then reinstall the remaining requirements if needed.

## Preprocessing Pipeline
The current preprocessing pipeline lives under `src/preprocessing/` and is run dataset-by-dataset.

Raw-to-interim parsers:
- `src/preprocessing/parse_sisfall.py`
- `src/preprocessing/parse_fallalld.py`
- `src/preprocessing/parse_umafall.py`
- `src/preprocessing/parse_upfall.py`

Shared interim-to-processed entrypoint:
```bash
python3 -m src.preprocessing.run_pipeline --dataset sisfall
```

Typical preprocessing workflow:
```bash
python3 src/preprocessing/parse_sisfall.py
python3 src/preprocessing/parse_fallalld.py
python3 src/preprocessing/parse_umafall.py
python3 src/preprocessing/parse_upfall.py

python3 -m src.preprocessing.run_pipeline --dataset sisfall
python3 -m src.preprocessing.run_pipeline --dataset fallalld
python3 -m src.preprocessing.run_pipeline --dataset umafall
python3 -m src.preprocessing.run_pipeline --dataset upfall
```

The current processing flow is:
1. Parse raw data into an interim trial pickle.
2. Build subject-wise train/val/test splits.
3. Attach splits to each trial.
4. Generate overlapping fixed-length windows.
5. Label windows with the fall-region overlap rule.
6. Save raw windows and metadata.
7. Fit normalization statistics per mode.
8. Export classification and TSAD-ready splits.

Current trial-level interim schema:
- `subject_id`
- `trial_id`
- `activity_id`
- `is_fall`
- `sampling_rate_hz`
- `n_samples`
- `acc`
- `raw_file`

Current processed window metadata schema:
- `window_id`
- `subject_id`
- `activity_id`
- `trial_id`
- `is_fall`
- `split`
- `sampling_rate_hz`
- `start_idx`
- `end_idx`
- `window_label`
- `tsad_train_eligible`

For full stage-by-stage schemas and saved artifact details, see [docs/preprocessing_pipeline.md](docs/preprocessing_pipeline.md). For earlier plans and design decisions, see [docs/preprocessing_legacy.md](docs/preprocessing_legacy.md).

## Usage
