import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from segment_label import segment_and_label
from run_stage_1 import butter_lowpass_filter, parse_filename
from load_signal import load_signal

# === CONFIG ===
DATA_DIR = Path("data/raw/sisfall")
SAMPLE_FILE = DATA_DIR / "SA01" / "D01_SA01_R01.txt"  # <-- pick any fall file
FS = 200  # Hz
WINDOW_SIZE = 200
STRIDE = 100

# === Load and preprocess one recording ===
meta = parse_filename(SAMPLE_FILE.name)
signals = load_signal(SAMPLE_FILE)

# Filter accelerometer data (use acc1)
acc = butter_lowpass_filter(signals["acc1"], cutoff=5, fs=FS)

# Compute and filter SMV
smv = np.sqrt((acc ** 2).sum(axis=1))
smv = butter_lowpass_filter(smv, cutoff=5, fs=FS)

# Segment and label
X, y = segment_and_label(acc, smv, meta, window_size=WINDOW_SIZE, stride=STRIDE)

# === Visualize ===
t = np.arange(len(smv)) / FS
impact_idx = np.argmax(smv)
fall_start = max(0, impact_idx - 100)
fall_end = min(len(smv), impact_idx + 100)

plt.figure(figsize=(14, 6))
plt.plot(t, smv, label="SMV (filtered)", color="black", lw=1)
plt.axvline(impact_idx / FS, color="red", ls="--", label="Impact peak")
plt.axvspan(fall_start / FS, fall_end / FS, color="orange", alpha=0.3, label="Fall region (±0.5s)")

# Mark fall windows
window_times = np.arange(0, len(smv) - WINDOW_SIZE, STRIDE) / FS
for i, start_t in enumerate(window_times):
    if y[i] == 1:
        plt.axvspan(start_t, start_t + WINDOW_SIZE / FS, color="green", alpha=0.3)

plt.title(f"{meta['activity']} | {meta['subject']} | Fall windows (green) vs Fall region (orange)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration magnitude (g)")
plt.legend()
plt.tight_layout()
plt.savefig("sisfall_fall_window_labeling.png", dpi=300)
