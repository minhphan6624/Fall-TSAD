#!/usr/bin/env bash
set -euo pipefail

# Scratch runbook for CNN/CNN-AE training commands.
# Run from the repository root after activating the project environment.

MODEL_SEED=42
N_FOLDS=5
DEVICE=auto

train_folds() {
  local prefix="$1"
  local trainer="$2"

  for fold in 0 1 2 3 4; do
    python -m "src.training.${trainer}" \
      --dataset "${prefix}_fold${fold}" \
      --model-seed "$MODEL_SEED" \
      --device "$DEVICE"
  done
}

aggregate_model() {
  local prefix="$1"
  local mode="$2"
  local model="$3"

  python -m src.training.aggregate_cv_metrics \
    --dataset-prefix "$prefix" \
    --n-folds "$N_FOLDS" \
    --mode "$mode" \
    --model "$model" \
    --model-seed "$MODEL_SEED"
}

# --------------------------------------
# Main 20 Hz 2s benchmark: CNN1D AE
# --------------------------------------

# for prefix in sisfall_20hz_2s fallalld_20hz_2s umafall_20hz_2s upfall_20hz_2s; do
#   train_folds "$prefix" train_cnn1d_ae
#   aggregate_model "$prefix" tsad cnn1d_ae
# done

# --------------------------------------
# Sampling-rate ablation: SisFall + FallAllD
# --------------------------------------

for prefix in sisfall_native_2s fallalld_native_2s; do
  train_folds "$prefix" train_cnn1d_large
  train_folds "$prefix" train_cnn1d_ae_large

  aggregate_model "$prefix" classification cnn1d_large
  aggregate_model "$prefix" tsad cnn1d_ae_large
done

for prefix in sisfall_20hz_2s fallalld_20hz_2s; do
  train_folds "$prefix" train_cnn1d_ae
  train_folds "$prefix" train_cnn1d_ae_large

  aggregate_model "$prefix" tsad cnn1d_ae
  aggregate_model "$prefix" tsad cnn1d_ae_large
done
