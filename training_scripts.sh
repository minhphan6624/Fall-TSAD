#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating the project environment.

MODEL_SEED=42
N_FOLDS=5
DEVICE=auto

train_folds() {
  local prefix="$1"
  local trainer="$2"

  for fold in 0 1 2 3 4; do
    python3 -m "src.training.${trainer}" \
      --dataset "${prefix}_fold${fold}" \
      --model-seed "$MODEL_SEED" \
      --device "$DEVICE"
  done
}

aggregate_model() {
  local prefix="$1"
  local mode="$2"
  local model="$3"

  python3 -m src.training.aggregate_cv_metrics \
    --dataset-prefix "$prefix" \
    --n-folds "$N_FOLDS" \
    --mode "$mode" \
    --model "$model" \
    --model-seed "$MODEL_SEED"
}

# Window-duration experiment: missing CNN1D and CNN1D-AE 3s/5s runs.
for prefix in \
  sisfall_20hz_3s sisfall_20hz_5s \
  fallalld_20hz_3s fallalld_20hz_5s \
  umafall_20hz_3s umafall_20hz_5s \
  upfall_20hz_3s upfall_20hz_5s; do
  train_folds "$prefix" train_cnn1d
  aggregate_model "$prefix" classification cnn1d

  train_folds "$prefix" train_cnn1d_ae
  aggregate_model "$prefix" tsad cnn1d_ae
done
