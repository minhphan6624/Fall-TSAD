# split.py
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import numpy as np

# CONFIG
META_CSV = "metadata.csv"            # produced by build_metadata.py
OUT_DIR  = Path("splits")            # where to write split CSVs
SEED     = 42

# how big the validation set is (fractions of available files)
VAL_ADL_FRAC  = 0.10   # % of young-ADL files to use in val
VAL_FALL_FRAC = 0.20   # % of young-fall files to use in val

# optional: subject-level holdout (set to a list like ["SA03","SA05"] to keep them entirely for TEST)
SUBJECT_HOLDOUT: List[str] = []


def load_meta(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["group"] = df["group"].replace({"Adult": "young", "Elderly": "elderly"})
    df["is_fall"] = df["is_fall"].astype(int)
    return df


def make_splits(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    # 1) Start from young cohort
    young = meta.query("group == 'young'").copy()

    # Optionally hold out some young subjects entirely for TEST (cross-subject evaluation)
    if SUBJECT_HOLDOUT:
        holdout_mask = young["subject"].isin(SUBJECT_HOLDOUT)
        young_holdout = young[holdout_mask]
        young = young[~holdout_mask]
    else:
        young_holdout = young.iloc[0:0]  # empty

    # 2) TRAIN = young ADL only
    train = young[(young["is_fall"] == 0)].copy()

    # 3) VAL = young ADL subset + young FALL subset
    # From the remaining young, draw fractions for validation
    val_adl_pool  = young[young["is_fall"] == 0]
    val_fall_pool = young[young["is_fall"] == 1]

    # Helper to sample a fraction of a DataFrame
    def _sample_frac(df: pd.DataFrame, frac: float) -> pd.DataFrame:
        if len(df) == 0 or frac <= 0:
            return df.iloc[0:0]
        
        n = max(1, int(round(frac * len(df))))
        idx = rng.choice(df.index, size=min(n, len(df)), replace=False)
        return df.loc[idx]

    val_adl  = _sample_frac(val_adl_pool,  VAL_ADL_FRAC)
    val_fall = _sample_frac(val_fall_pool, VAL_FALL_FRAC)
    val = pd.concat([val_adl, val_fall], axis=0).drop_duplicates()

    # Ensure VAL files are not in TRAIN
    train = train[~train["path"].isin(val["path"])]

    # 4) TEST = everything not in TRAIN or VAL,
    #          plus ALL elderly (ADL + FALL) and young holdout subjects
    used_paths = set(pd.concat([train, val], axis=0)["path"].tolist())
    rest_young = young[~young["path"].isin(used_paths)]
    elderly    = meta.query("group == 'elderly'").copy()
    test = pd.concat([rest_young, elderly, young_holdout], axis=0).drop_duplicates()

    # sanity checks
    assert set(train["path"]).isdisjoint(set(val["path"]))
    assert set(train["path"]).isdisjoint(set(test["path"]))
    assert set(val["path"]).isdisjoint(set(test["path"]))

    return train.sort_values("path"), val.sort_values("path"), test.sort_values("path")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta(META_CSV)

    train, val, test = make_splits(meta)

    # Save split files (CSV with same columns as metadata)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    # Small report
    def brief(df, name):
        n_files = len(df)
        n_fall  = int(df["is_fall"].sum())
        n_adl   = n_files - n_fall
        print(f"{name:<5} | files={n_files:4d}  ADL={n_adl:4d}  FALL={n_fall:4d}  "
              f"subjects={df['subject'].nunique():2d} (young={sum(df['group']=='young'):4d}, elderly={sum(df['group']=='elderly'):4d})")

    brief(train, "TRAIN")
    brief(val,   "VAL")
    brief(test,  "TEST")


if __name__ == "__main__":
    main()
