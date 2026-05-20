# Easy-ADL Experiment

This experiment tests whether training with high-motion or fall-like ADLs is hurting the current models.

The main comparison should keep validation and test sets unchanged. Only normal training windows are filtered to the selected easy ADLs.

For classification models:

- keep all fall training windows
- keep only normal training windows whose `activity_id` is in the easy-ADL list

For TSAD models:

- train only on normal windows whose `activity_id` is in the easy-ADL list

Normalization should be fit on the filtered training data used by each mode.

## Easy ADL Lists

The canonical machine-readable list is stored in `configs/easy_adl_activities.json`.

### SisFall

| ID | Activity |
| --- | --- |
| `D07` | Slowly sit in a half-height chair, wait, and stand up slowly |
| `D09` | Slowly sit in a low-height chair, wait, and stand up slowly |
| `D12` | Sit, lie slowly, wait, and sit again |
| `D14` | Turn in bed |
| `D15` | Slowly bend at knees and stand up |
| `D16` | Slowly bend without bending knees and stand up |

Excluded examples: jogging, quick stairs, stumble, jump, quick sit/stand, and collapse-into-chair activities.

### FallAllD

| ID | Activity |
| --- | --- |
| `A013` | Sitting down |
| `A014` | Standing up |
| `A028` | Bending down and raising up |
| `A037` | Start ascending using a lift |
| `A038` | Stop ascending using a lift |
| `A039` | Start descending using a lift |
| `A040` | Stop descending using a lift |

Current parsed waist-device ADLs start at `A013`; `A001-A012` are listed in the mapping PDF but are not present in the current parsed D3 ADL files.

Excluded examples: walking, stairs, jogging, jumping, stumbling, bed transfers, and fail-to-stand.

### UMAFall

| ID | Activity |
| --- | --- |
| `Aplausing` | Applauding |
| `MakingACall` | Making a call |
| `HandsUp` | Hands up |
| `Bending` | Bending |
| `OpeningDoor` | Opening a door |

Excluded examples: walking, jogging, hopping, stairs, lying down, and chair sit/get-up.

### UP-FALL

| ID | Activity |
| --- | --- |
| `7` | Standing |
| `8` | Sitting |
| `9` | Picking up object |
| `11` | Laying |

Excluded examples: walking and jumping.

## Dataset Generation Steps

1. Add an easy-ADL filter to the processed dataset generation path.
   - The filter should apply only to training windows.
   - Validation and test exports should remain unchanged.
   - `activity_id` should be used as the filter key.
   - Pass the filter with `--adl-filter-config configs/easy_adl_activities.json`.

2. Preserve the normal benchmark settings.
   - `target_sampling_rate_hz=20`
   - `window_seconds=2`
   - `overlap=0.5`
   - `split_protocol=subject_kfold`
   - `n_folds=5`
   - same split seed and fold assignments as the main benchmark

3. Generate explicit easy-ADL CV dataset variants.
   - Recommended prefixes:
     - `sisfall_easy_adl_20hz_2s`
     - `fallalld_easy_adl_20hz_2s`
     - `umafall_easy_adl_20hz_2s`
     - `upfall_easy_adl_20hz_2s`
   - Fold outputs should be named `<prefix>_fold0` through `<prefix>_fold4`.

4. Before full training, inspect each generated fold.
   - Count selected normal training windows.
   - Count fall training windows for classification.
   - Confirm validation and test activity distributions match the full-ADL baseline variant.

5. Train the same model set used in the main benchmark.
   - Classification: Random Forest, XGBoost, CNN1D, LSTM classifier.
   - TSAD: Isolation Forest, Dense AE, CNN1D AE, LSTM AE.

6. Aggregate CV results with the existing aggregator.
   - Use the easy-ADL dataset prefix.
   - Compare against the matching full-ADL `*_20hz_2s` CV results.

7. Record results in `docs/results_trackers/easy_adl.csv` and generate the report table after all folds are complete.

## Implementation Notes

Save the selected easy-ADL list into the processed artifact or each run `config.json`. The experiment should be auditable without relying on this document alone.

If the filter is implemented in preprocessing, the mode exports need to fit normalizers after filtering. If the filter is implemented in training, the training scripts need to save the filter list and ensure their normalizers or loaded tensors reflect the filtered train set.

## Result Interpretation Notes

The easy-ADL training variant changes the training distribution only. Validation and test still contain all ADLs and falls.

For classification models, this means:

- training positives stay as all fall windows
- training negatives become easy normal ADL windows only
- held-out negatives still include hard ADLs

For TSAD models, this means:

- the learned normal region is based only on easy normal ADL windows
- hard ADLs in validation/test are expected to look more anomalous

This setup can reduce training confusion, but it can also remove important hard negative examples. If precision or AUPRC drops, a likely explanation is that hard ADLs are being ranked as fall-like because the model no longer saw them as normal during training.

Observed shallow-model behavior so far follows this pattern:

- SisFall false positives concentrate in excluded activities such as `D04` jogging quickly, `D03` jogging slowly, and `D06` quick stairs.
- FallAllD false positives concentrate in excluded jogging and fast stair activities such as `A025`, `A044`, `A024`, `A036`, and `A043`.
- UMAFall false positives concentrate in excluded `Hopping` and `Jogging`.
- Isolation Forest is especially sensitive because easy-ADL training makes the normal region narrow; it can learn "different from easy ADLs" rather than "fall-like."

Some false positives may appear under fall activity IDs. These are normal-labeled windows from fall trials outside the labeled fall region. They still affect window-level metrics, but they are not ADL false alarms in the usual sense.

The main conclusion to test is not simply whether easy-ADL training improves F1. The more useful question is whether removing hard normal ADLs improves fall ranking and detection on the same full held-out ADL/fall distribution. A drop in AUPRC is stronger evidence than a drop in thresholded precision because AUPRC reflects score ranking, not just the selected validation threshold.

## Easy-Only Evaluation Variant

Do not remove hard ADLs from validation/test for the main easy-ADL experiment. That would change the task from realistic held-out fall detection to a simplified normal-distribution benchmark.

However, an easy-only evaluation can be justified as a separate diagnostic experiment:

```text
Train, validate, and test on a restricted low-motion ADL subset to measure model behavior under a simplified normal-activity distribution.
```

This answers:

```text
If the deployment environment only contains simple low-motion ADLs, how well do models separate falls from easy normal behavior?
```

It does not answer:

```text
How well does the model work for realistic fall detection across diverse ADLs?
```

Valid uses of an easy-only variant:

- controlled distribution study
- best-case or upper-bound comparison
- deployment-specific scenario where high-motion ADLs are not expected
- diagnostic check to see whether models still fail when hard ADL confusion is removed
- possible curriculum-learning baseline

Important caveats:

- it will likely inflate precision, specificity, F1, and AUPRC relative to full-ADL evaluation
- it underestimates false alarms in realistic settings
- it changes the benchmark task definition
- it is less comparable to the main full-ADL benchmark
- it can remove clinically relevant hard non-falls such as stairs, stumbling, or quick sit/stand

If implemented, use a distinct name such as:

```text
<dataset>_easy_adl_only_20hz_2s
```

Recommended reporting structure:

```text
Full-ADL benchmark: realistic evaluation.
Easy-train/full-test: tests whether easy normal training generalizes.
Easy-only benchmark: restricted-normal upper bound.
```
