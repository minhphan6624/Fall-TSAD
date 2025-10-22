# 🧭 SisFall Data Splitting & Preprocessing Guide
**Project:** Fall Detection via Time-Series Anomaly Detection (TSAD)  
**Goal:** Train only on *normal ADL* patterns (young participants) and detect *falls* as anomalies.  

---

## 🎯 Overview
This pipeline prepares the **SisFall dataset** for anomaly-based fall detection.  
It ensures **no leakage**, proper **normal-only training**, and a **partially labeled validation set** for threshold calibration.

---

## 1️⃣ Dataset Splitting Strategy

| Split | Contains | Purpose |
|:--|:--|:--|
| **Train** | ADL (D01–D19) from **young** participants | Learn normal motion patterns |
| **Validation** | ADL + small % FALLs from **young** participants | Threshold sweeping (F1 / FAR) |
| **Test** | ADL + FALLs from **young + elderly** | Final performance & generalization |

### Notes
- Split **by file**, not by window.  
- File overlap between splits is **not allowed**.  
- **Subject-wise split** is *optional*:  
  <!-- - ✅ Use if you want to evaluate **cross-subject generalization**.   -->
  - ⚙️ Omit if you assume per-user calibration (more realistic for wearables). (Probably omit for now)

---

## 2️⃣ Preprocessing Steps (in correct order)

### Step 1 — Build Metadata
Parse file names → create a table with:  
`[path, code, subject, group, is_fall]`  
- `group`: `"young"` for SAxx, `"elderly"` for SExx  
- `is_fall`: `1` if code starts with `"F"`

### Step 2 — Split by File
- **Train:** young ADL only.  
- **Val:** young ADL + small % of young FALLs.  
- **Test:** remaining young ADL/FALL + all elderly ADL/FALL.  
- Save split lists (`train.csv`, `val.csv`, `test.csv`).

### Step 3 — Load & Filter Signals
- Use **ADXL345 accelerometer** (3 axes).  
- Sampling rate: **200 Hz**.  
- Apply **4th-order Butterworth LPF (5 Hz)**.  
- Optionally clip |acc| > 8 g.

### Step 4 — Normalize
- Use **RobustScaler (median/IQR)**.  
- Fit on **train ADL** data only.  
- Apply same transform to val/test.

### Step 5 — Segment
- **Window length:** 3 s (≈ 600 samples @ 200 Hz)  
- **Overlap:** 50 % (≈ 1.5 s stride)  
- Segment **within each file** separately.

---

## 3️⃣ Window Labeling

| File Type | Label Rule |
|:--|:--|
| **ADL** | All windows → 0 |
| **FALL** | Label 1 only for windows overlapping impact zone |

### Impact zone detection
1. Compute magnitude `|acc| = √(ax² + ay² + az²)`  
2. Find peak `t*` (impact).  
3. Label as 1 if window overlaps `[t* − 0.5 s, t* + 1.0 s]`.  
4. Others → label 0 (or discard).

---

## 4️⃣ Thresholding Strategy

After training your model (e.g., LSTM-AE):

1. Compute reconstruction errors on **validation set (ADL + FALL)**.  
2. Sweep thresholds τ:  
   - **Option A:** maximize F1 (precision–recall balance).  
   - **Option B:** minimize False-Alarm Rate (FAR) subject to Recall ≥ target.  
3. Save best τ and apply on test set.

---

## 5️⃣ Evaluation

| Level | Description |
|:--|:--|
| **Window-level** | Treat each window as a sample. |
| **Event-level** | A fall is detected if *any* of its windows ≥ τ. |

**Metrics:** Precision, Recall, F1, False-Alarm Rate.  
Include **elderly ADL** in test to measure robustness.

---

## ✅ Summary Checklist

- [x] Split by **file** before any preprocessing  
- [x] Train = **young ADL** only  
- [x] Val = **young ADL + small FALL subset**  
- [x] Test = **young + elderly ADL + FALL**  
- [x] Fit scaler on train ADL only  
- [x] Segment 3 s windows (50 % overlap)  
- [x] Label falls ± 0.5 s/1 s around impact  
- [x] Sweep threshold on validation (F1 or FAR)  
- [x] Evaluate on test (window & event level)

---

## 📁 Example file outputs
```
data/
  processed/
    sisfall/
      metadata.csv
      splits/
        train.csv
        val.csv
        test.csv
      train_data.npy
      val_data.npy
      test_data.npy
```

---

## 🧩 Notes
- Always verify that train/val/test sets are **disjoint by file**.  
- If using subject-wise disjoint split, make that clear in results.  
- This pipeline works identically for other TSAD models (AE, VAE, MSCRED, TranAD, etc.).  
