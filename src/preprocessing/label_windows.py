from __future__ import annotations

import numpy as np
import pandas as pd


FALL_REGION_HALF_WIDTH_SECONDS = 1.0
FALL_LABEL_THRESHOLD = 0.60


def compute_trial_label_info(trial_row: pd.Series) -> dict[str, float | int]:
    ''' Compute 3 main fall-related info from a trial
    Args:
    - trial_row: a row with necessary data for an activity trial

    Output: A dictionary containing key & value pairs for:
        - Impact index (peak SMV value)
        - fall_start: the index at which the fall region starts
        - fall_end: the index at which the fall region ends

    '''
    
    if int(trial_row["is_fall"]) == 0:
        return {
            "impact_idx": -1,
            "fall_start": -1,
            "fall_end": -1,
        }

    # ----- Find impact index (Peak SMV) and fall region -----
    acc = np.asarray(trial_row["acc"], dtype=np.float32)
    
    smv = np.sqrt(np.sum(acc * acc, axis=1))
    impact_idx = int(np.argmax(smv))
    
    half_width = int(round(float(trial_row["sampling_rate_hz"]) * FALL_REGION_HALF_WIDTH_SECONDS))
    start_idx = max(0, impact_idx - half_width)
    end_idx = min(acc.shape[0], impact_idx + half_width)

    if end_idx <= start_idx:
        end_idx = min(acc.shape[0], start_idx + 1)

    return {
        "impact_idx": impact_idx,
        "fall_start": int(start_idx),
        "fall_end": int(end_idx),
    }


def overlap_ratio(
    window_start: int, window_end: int,
    fall_start: int, fall_end: int,
) -> float:
    
    if fall_start < 0 or fall_end <= fall_start:
        return 0.0

    overlap = max(0, min(window_end, fall_end) - max(window_start, fall_start))
    fall_region_length = fall_end - fall_start
    
    return float(overlap) / float(fall_region_length)


def label_windows(trials_df: pd.DataFrame, window_meta_df: pd.DataFrame) -> pd.DataFrame:
    computed_rows: list[dict[str, float | int]] = []

    # COmpute and add the trial fall-related info with the trial df
    for row in trials_df.itertuples(index=False):
        label_info = compute_trial_label_info(pd.Series(row._asdict()))
        label_info["subject_id"] = row.subject_id
        label_info["activity_id"] = row.activity_id
        label_info["trial_id"] = row.trial_id
        computed_rows.append(label_info)

    label_info_df = pd.DataFrame(computed_rows)
    labeled = window_meta_df.merge(
        label_info_df,
        on=["subject_id", "activity_id", "trial_id"],
        how="left",
    )


    fall_overlap_ratio = labeled.apply(
        lambda row: overlap_ratio(
            int(row["start_idx"]),
            int(row["end_idx"]),
            int(row["fall_start"]),
            int(row["fall_end"]),
        ),
        axis=1,
    )

    # Window is fall if it's a fall trial and the overlap ratio is greater than the threshold
    labeled["window_label"] = (
        (labeled["is_fall"].astype(int) == 1) & (fall_overlap_ratio > FALL_LABEL_THRESHOLD)
    ).astype(np.int64)
    
    labeled["tsad_train_eligible"] = (
        (labeled["split"] == "train") & (labeled["window_label"] == 0)
    ).astype(np.int64)
    
    return labeled.drop(columns=["impact_idx", "fall_start", "fall_end"])
