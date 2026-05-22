# Cross-Validation Results Observations

This note records interpretation points from the current 20 Hz, 2 second,
subject-wise 5-fold CV benchmark results under `runs/benchmark`.

The goal is not to replace the metric trackers. Use the aggregated
`cv_metrics.csv` files for exact mean/std values. This document records the
main conclusions and how to reason about them.

## Benchmark Context

- Windowing: 2 second windows at 20 Hz.
- Evaluation: subject-wise 5-fold cross-validation.
- Threshold policy: thresholds are selected on the validation fold and applied
  to the test fold.
- Metrics are window-level metrics.
- Classification models train on labeled fall and non-fall windows.
- TSAD models train as anomaly detectors, generally using normal training
  windows and then scoring test windows by anomaly/reconstruction score.

## Main Conclusions

Tree-based supervised classifiers are the strongest practical fall detectors
in the current results. XGBoost is the most consistent top performer across
UMAFall, FallAllD, and UP-FALL. Random Forest is usually second and often close,
but it shows more fold-to-fold false-positive variation.

The supervised deep classifiers do not clearly beat the simpler feature-based
tree models. CNN1D is often reasonable, but it trails XGBoost and Random Forest
on F1 and precision/recall balance. The LSTM classifier is generally weaker in
these current runs. For these small wearable datasets and short windows,
engineered features plus tree models appear to be a strong baseline.

The TSAD models often have high AUROC, but much lower AUPRC, precision, and F1.
This means the anomaly scores contain useful ranking signal, but the score
distributions are not separated cleanly enough to make a reliable binary
detector at the selected threshold.

AUROC should not be used alone to judge fall-detection usefulness. Fall windows
are rare, so a model can rank positives above negatives fairly well overall
while still producing too many false positives at any practical threshold.
AUPRC, precision, recall, specificity, F1, and the FP/FN counts give a more
realistic picture.

## Classification vs TSAD

The main difference is calibration and operating-point quality.

Classification models learn directly from fall and non-fall examples. They are
therefore better aligned with the final binary detection task.

TSAD models learn what normal windows look like and treat deviations as
possible falls. This is useful when fall labels are scarce, but high-motion ADLs
can also look anomalous. As a result, TSAD often catches many falls but also
flags too many normal windows.

High TSAD AUROC is not necessarily bad. It says the score ranking is useful.
The problem is that deployment-style fall detection requires choosing a single
threshold, and the current thresholds often produce either too many false
positives or too many missed falls.

## Why High AUROC Can Coexist With Low F1

AUROC evaluates ranking over all possible thresholds. F1 evaluates one chosen
threshold.

In imbalanced fall detection, there are many more normal windows than fall
windows. Even a modest false-positive rate can create many false alarms, which
pushes precision and F1 down. This is why TSAD models can show AUROC around
`0.9+` while still having weak precision and F1.

AUPRC is more informative than AUROC for this task because it focuses on how
well positive fall windows are concentrated near the top of the score ranking.
When AUROC is high but AUPRC is much lower, the model has some signal, but not
enough clean separation for reliable alerting.

## Fold-Level Interpretation

Subject-wise CV is more realistic than window-wise splitting because every test
fold contains unseen subjects. It also makes folds naturally uneven.

Some folds contain subjects with many more trials/windows than others. Raw
counts such as TN and TP can therefore have large standard deviations even when
the model's rates are stable. Use specificity, recall, precision, and F1 to
judge performance stability. Use raw FP/FN counts to understand operational
impact.

Hard folds are expected. A fold can be harder because:

- held-out subjects move differently from training subjects
- fall styles differ by subject
- high-energy ADLs resemble falls
- a fold has few fall windows, so each missed fall has large metric impact
- validation-selected thresholds do not transfer perfectly to test subjects
- some subjects contribute many more windows than others

## Easy-ADL Training Observations

Early shallow-model results from the easy-ADL training variants show a useful
failure mode: restricting normal training windows to easy ADLs can reduce
precision and AUPRC even when AUROC remains high.

This happens because the training distribution changes, but validation and test
remain full-ADL. The models no longer see hard normal ADLs as negative examples
during training, so excluded high-motion ADLs are more likely to be scored as
fall-like at test time.

Observed false positives concentrate in the expected excluded activities:

- SisFall: `D04` jogging quickly, `D03` jogging slowly, and `D06` quick stairs.
- FallAllD: jogging and fast stair activities such as `A025`, `A044`, `A024`,
  `A036`, and `A043`.
- UMAFall: `Hopping` and `Jogging`.

This supports the interpretation that hard ADLs are not merely noisy training
examples. They are important hard negatives that help supervised classifiers
learn what fall-like non-falls look like.

For Isolation Forest, the effect is stronger because the model learns a narrow
normal region from easy ADLs only. It can become good at detecting "different
from easy ADL" rather than "fall-like," which increases false positives on hard
ADLs.

A drop in precision alone could be an operating-threshold issue. A drop in
AUPRC is more important because it means the score ranking itself is worse:
hard ADL windows are being ranked too close to, or above, true fall windows.

The main easy-ADL experiment should keep validation and test unchanged. Removing
hard ADLs from validation/test would define a separate easy-only benchmark. That
can be reported as a restricted-normal upper bound, but it should not replace
the full-ADL evaluation because it hides the realistic false-positive burden.

## UMAFall Observations

On UMAFall 20 Hz 2s, XGBoost is the strongest model overall. Random Forest is
close but less stable, with larger false-positive variation on some folds.
CNN1D ranks windows fairly well but does not match the tree models on
thresholded detection metrics. The LSTM classifier is weaker.

The TSAD models have high AUROC but much lower AUPRC and F1. They are better
interpreted as weak-to-moderate anomaly rankers than reliable calibrated fall
detectors.

Fold effects are visible. Fold 1 has many more test windows because Subject_18
contributes a large number of windows. Fold 2 is harder for recall because it
has relatively few fall windows and several missed falls from held-out subjects
such as Subject_14 and Subject_17. Random Forest also shows false positives on
high-motion activities such as hopping and jogging.

## FallAllD Observations

On FallAllD 20 Hz 2s, XGBoost and Random Forest are again the strongest models.
XGBoost has the best overall balance in the current CV aggregate, while Random
Forest is close.

FallAllD is highly imbalanced at the window level, with fall windows around
roughly one to two percent of each test fold. This makes precision hard and
makes AUPRC/F1 more important than AUROC.

TSAD models again show high AUROC but weak AUPRC and F1. Dense autoencoder is
the strongest TSAD model among the current FallAllD results, but it is still far
behind the supervised tree models as a binary detector.

Fold 0 appears to be a difficult fold across multiple models. Both supervised
tree models lose recall on this fold, and the TSAD models degrade more sharply.
This suggests a subject/fold distribution issue rather than a single-model
failure.

## UP-FALL Observations

On UP-FALL 20 Hz 2s, XGBoost is very strong and clearly the best current model.
Random Forest is second, with strong recall but more false-positive variation.
CNN1D is weaker, and the LSTM classifier is much weaker in the current runs.

The TSAD models on UP-FALL show very high recall but very low precision. This
means they catch many fall windows by flagging too many normal windows. Their
average false-positive counts are much higher than the supervised models.

This is a classic anomaly-detection failure mode for fall detection: the model
is sensitive to abnormal or high-motion windows, but many non-fall activities
also look abnormal enough to cross the threshold.

## Practical Reporting Guidance

For final result tables, emphasize:

- F1 for overall thresholded detection quality
- precision and FP counts for false-alarm burden
- recall and FN counts for missed-fall burden
- AUPRC for ranking quality under class imbalance
- AUROC as a secondary ranking metric, not the main conclusion

When discussing TSAD, phrase the result carefully:

> The TSAD models learn useful anomaly scores, as reflected by high AUROC, but
> they are not well calibrated as binary fall detectors under the current
> threshold policy. Their lower AUPRC, precision, and F1 indicate substantial
> score overlap between falls and normal high-motion activities.

When discussing fold variation, avoid over-interpreting raw TN/TP standard
deviations. Large raw-count std can come from uneven fold sizes. Check
normalized metrics and fold-level FP/FN patterns before claiming instability.
