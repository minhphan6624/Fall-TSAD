from pathlib import Path

raw_files = list(Path("data/raw/sisfall").rglob("*.txt"))
# remove readme
raw_files = [f for f in raw_files if f.name.lower() != "readme.txt"]
processed_files = list(Path("data/processed/sisfall/stage_1").glob("*.npz"))

print(f"Raw: {len(raw_files)}, Processed: {len(processed_files)}")

if len(raw_files) == len(processed_files):
    print("✅ All files processed successfully!")
else:
    print(f"⚠️ Missing {len(raw_files) - len(processed_files)} files.")

import numpy as np, random

sample_file = random.choice(processed_files)
data = np.load(sample_file, allow_pickle=True)

for key in ['acc1', 'gyro', 'acc2']:
    arr = data[key]
    print(key, arr.shape, arr.dtype)

meta = data["metadata"].item()
print(meta)

import matplotlib.pyplot as plt

acc = data["acc1"]
plt.figure(figsize=(10,4))
plt.plot(acc[:, 0], label='X')
plt.plot(acc[:, 1], label='Y')
plt.plot(acc[:, 2], label='Z')
plt.legend()
plt.title(f"Accelerometer (filtered) – {meta['activity']} {meta['subject']}")
plt.show()

from load_signal import load_signal
from filter import butter_lowpass_filter

raw = load_signal(Path("data/raw/sisfall/SA01/F05_SA01_R01.txt"))
filtered = butter_lowpass_filter(raw["acc1"], cutoff=5, fs=200)

plt.plot(raw["acc1"][:,0], color="gray", alpha=0.5, label="Raw")
plt.plot(filtered[:,0], color="blue", label="Filtered")
plt.legend()
plt.title("Before vs After Filtering")
plt.show()

