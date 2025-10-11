# Overview

# Folder structure
* config
* * 

# How to run

## Prerequisites
Python .= 3.10
PyTorch

## Data preparation pipeline
```python
python -m scripts.preprocess
```

## Training script
python -m scripts.train

## Select threshold
```python
python src/modeling/select_threshold.py experiment_dir=outputs/2025-10-11/12-29-00/
```

## Evaluate model
```python
python src/modeling/evaluate.py experiment_dir=outputs/2025-10-11/12-29-00/
```

