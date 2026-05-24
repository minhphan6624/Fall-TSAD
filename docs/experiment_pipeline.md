# Benchmark Experiment Pipeline

This document summarizes the experiment-level decisions for the benchmarking study comparing time-series anomaly detection (TSAD) and supervised classification methods for wearable fall detection.

It supersedes the earlier SisFall-only anomaly-detection notes.

## Goal

Build a fair benchmark across four datasets after they are converted into the shared interim trial format.

The benchmark should isolate model-family differences rather than preprocessing differences.

## Core Benchmark Principle

The most important next step after raw parsing is the shared processed pipeline, not adding more models immediately.

The benchmark should keep the following components identical across TSAD and classification:

- dataset inclusion
- sensor channel selection
- subject-wise split definitions
- window length and stride
- labeling policy
- normalization policy
- evaluation protocol

Only the learning setup should differ:

- TSAD trains on normal windows only
- classification trains on both fall and non-fall windows

## Shared Pipeline Priorities

1. subject-wise split builder
2. common windowing pipeline
3. common labeling pipeline
4. common normalization pipeline
5. processed export and evaluation plumbing

This order matters because split and labeling policy determine whether the comparison is methodologically valid.

## Split Policy

SUbject-based splitting

Subject-disjoint for 

However, for final result, use LEave one subject out, or even k-fold validatiaon with k=5 for evaluation base

## Windowing Policy

The shared processed layer should turn each trial-level `acc` array into fixed-length windows plus metadata.

Per-window metadata and other setup can be found in `docs/preprocessing_pipeline.md`.

Current setup:

window = 2s in length (seconds)

## Labeling Policy

The main downstream target for the benchmark is fall vs non-fall, not multiclass activity recognition.

Implications:

- `is_fall` is the core benchmark label
- `activity_id` is retained mainly for provenance, inspection, debugging, and optional later grouping
- activity enumeration is not required now unless a later analysis specifically needs grouped activity labels

For the processed benchmark layer:

- define one explicit binary fall label per window
- use the same label rule for evaluating both TSAD and classification
- keep the label policy independent from model family

## Normalization Policy

Normalization must be fit on training data only.

Recommended default:

- fit per-axis statistics on training windows
- reuse the same transform on validation and test windows
- keep normalization identical across compared methods unless a deviation is explicitly justified

Category-specific application:

- TSAD: fit on normal training windows only
- classification: fit on all training windows

## Evaluation Priorities

Window-level metrics alone are not enough for a strong fall-detection benchmark.

- Precision
- Recall
- F1
- Specificity, 
- AUROC
- AUPRC

- use window-level metrics for core model comparison
- add trial-level or event-level evaluation later if the window-to-event mapping becomes stable enough to support it

## Experiment Management

Every run should be defined by a clear experiment manifest or equivalent config bundle containing:

- dataset
- split seed or split definition id
- window parameters
- channel selection
- label policy
- normalization policy
- model family
- model hyperparameters
- training configuration
- evaluation configuration

This is necessary so TSAD and classification runs remain comparable and reproducible.
