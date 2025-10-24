# SisFall Data Splitting & Preprocessing Guide
**Project:** Fall Detection via Time-Series Anomaly Detection (TSAD)  
**Goal:** Train only on *normal ADL* patterns (young participants) and detect *falls* as anomalies.  

---

## How to run the pipeline
From root:
```python
  python3 src/preprocessing/sisfall/run_pipeline.py
```
Then
```python
  python3 src/preprocessing/sisfall/split.py
```
FInally
```python
  python3 src/preprocessing/sisfall/normalize_dataset.py
```
## 1️⃣ Dataset Splitting Strategy

| Split | Contains |
|:--|:--|
| **Train** | ADL (D01–D19) from **young** participants (SA01-SA15)|
| **Validation** | ADL from **young** participants (SA16-SA17) | 
| **Test** | ADL + FALLs from **young + elderly** (SA18-SA23 + SE01-SE15)| 

### Notes
- Split **by file**, not by window.  
- File overlap between splits is **not allowed**.  
- **Subject-wise split** is *optional*:  
  <!-- - ✅ Use if you want to evaluate **cross-subject generalization**.   -->
  - ⚙️ Omit if you assume per-user calibration (more realistic for wearables). (Probably omit for now)

---

## 2️⃣ Preprocessing Steps (in correct order)

### Step 1 — Load Metadata from filename
- Parse filename to extract **metadata**
- Read and convert signal data to appropriate 

### Step 2 — Load & Filter Signals
- Use **ADXL345 accelerometer** and **gyroscope**
- Apply **4th-order Butterworth LPF (20 Hz)**.  (Currently not in use)
- Sampling rate: **200 Hz**.  

### Step 4 — per-sensor Normalization
- Use Robust Scaler to normalize each sensor reading seperately to main 

### Step 5 — Segment and label
- **Window length:** 1 s (≈ 200 samples @ 200 Hz)  
- **Overlap:** 50 % (100 sample step)  
---

## 3️⃣ Window Labeling

| File Type | Label Rule |
|:--|:--|
| **ADL** | All windows → 0 |
| **FALL** | Label 1 only for windows overlapping impact zone |

### Impact zone detection
1. Compute magnitude `|acc| = √(ax² + ay² + az²)`  
2. Find peak `t*` (impact).  
3. Windows from -0.5s to +0.5s around impact are labelled as fall.
4. Others → label 0 (or discard).

---

## 4️⃣ Thresholding Strategy
### Unsupervised

- Percentile cutoff: e.g. recon_error > 95% of recon_error dist --> anomalies
- mean + k.std: e.g 

### Supervised
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