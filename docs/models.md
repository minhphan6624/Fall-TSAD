# Model Selection Notes

This document summarizes the current model-selection decisions for the fall-detection benchmark comparing time-series anomaly detection (TSAD) and supervised classification.

## Selection Principle

The study should start with a small, defensible baseline set rather than a large collection of loosely related models.

Models should be chosen to cover clearly different inductive biases while keeping implementation and tuning effort comparable across categories.

Include models that:

- Are realistic to train on all four datasets
- Rre compatible with the same shared preprocessing pipeline
- Represent genuinely different modeling families
- can be tuned under a comparable validation budget

Avoid adding models only because they are fashionable or because they differ by a small architectural variant.

## Classification Baselines

- Feature-based classical baseline: Random Forest or XGBoost 
- 1D CNN classifier
- LSTM classifier

Rationale:

- the classical baseline provides a non-deep reference point
- the 1D CNN provides a strong discriminative temporal baseline
- the LSTM provides a sequence-modeling baseline

This set is enough to compare classical supervised learning against common deep discriminative approaches without over-expanding the study.

## TSAD Baselines

- Dense AE
- LSTM-AE
- VAE/LSTM-VAE

Rationale:

- these are standard reconstruction-based anomaly-detection baselines
- they are easier to position and interpret on wearable accelerometer windows
- they fit the current project scope better than more specialized long-horizon industrial TSAD models

## Transformer-Based TSAD

Transformer-based TSAD models are not excluded, but they are advanced comparison models rather than first-wave baselines.

- add a transformer TSAD model only after the shared benchmark pipeline is stable
- Focus on TranAD first, then Anomaly Transformer

Reasoning:

- the fall datasets are relatively small and low-dimensional compared with many industrial multivariate TSAD benchmarks
- transformer TSAD models add tuning sensitivity and can be harder to interpret fairly in a first-pass benchmark
- they are better framed as extension models once the benchmark protocol is already working

## Representation Learning Models

Representation-learning approaches are valid, but they broaden the scope of the study.

They should be considered only if the study explicitly wants a third category beyond:

- supervised classification
- reconstruction-based TSAD

Until then, they are better treated as optional follow-up experiments rather than part of the initial benchmark core.

## Fairness Rules Across 

To keep the comparison defensible:

- use the same input channels
- use the same subject split logic
- use the same window generation
- use the same normalization pipeline unless a deviation is justified and documented
- keep tuning effort comparable across model families
- report results under the same evaluation protocol

## Current Recommended Initial Roster

Minimal benchmark roster:

- Classification: classical baseline, 1D CNN, LSTM classifier
- TSAD: autoencoder, LSTM autoencoder, optional VAE

Optional extension (after the pipeline is stable)

- TSAD: TranAD, Anomaly Transformer

