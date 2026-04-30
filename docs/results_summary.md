# Results Summary

This document is for curated final or checkpoint results only.

Generated run artifacts under `runs/` are not version controlled. When a result is important enough to keep in the repository, copy the key metrics into this document instead of committing the full run folder.

## Current Status

No final benchmark results have been selected yet.

The current shallow training scripts have only been smoke-tested on development artifacts. Those smoke runs are useful for checking that the pipeline works, but they should not be treated as final results.

## Result Table Template

Use this table for selected benchmark results.

| Dataset | Split protocol | Sampling setup | Mode | Model | Model seed | AUROC | AUPRC | Precision | Recall | Specificity | F1 | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TBD | TBD | TBD | classification | Random Forest | 42 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## What To Record

For each result, record:

- dataset
- split protocol, such as development percentage split, k-fold, or leave-one-subject-out
- sampling setup, such as native-rate or downsampled 20 Hz
- learning mode, either classification or TSAD
- model name
- model seed, if the model has randomness
- threshold policy, currently validation best-F1
- AUROC
- AUPRC
- Precision
- Recall / Sensitivity
- Specificity
- F1
- any important caveats

## What Not To Record Here

Do not paste full prediction tables, model checkpoints, TensorBoard logs, or generated plots into this document.

Those should stay in local `runs/` folders or be archived outside git if needed.
