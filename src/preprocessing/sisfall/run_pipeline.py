from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

from load_signal import load_signal
from parse_filename import parse_filename
# from filter import butter_lowpass_filters
from normalize_sensor import normalize_sensor
from segment_label import segment_and_label

RAW_DIR = Path("data/raw/sisfall/")
OUT_DIR = Path("data/processed/sisfall/windows")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_trial(file_path):
    """Process a single trial file: load, normalize, and save."""
    try:

        # Load metadata and signals (converted)
        meta = parse_filename(file_path.name)
        signals = load_signal(file_path)

        # Normalize per-sensor (using RobustScaler)
        normed = {
            name: normalize_sensor(data, scaler=RobustScaler())
            for name, data in signals.items()
        }

        acc_data = normed['acc1']
        gyro_data = normed['gyro']
        combined_data = np.hstack([acc_data, gyro_data]) 

        # Compute SMV from normalized accelerometer data
        smv = np.sqrt(np.sum(acc_data**2, axis=1))

        # Segment & label
        X, y = segment_and_label(combined_data, smv, meta, window_size=200, stride=100)

        #6. Save windows + labels
        out_filename = f"{meta['activity']}_{meta['subject']}_{meta['trial']}.npz"
        np.savez_compressed(
            OUT_DIR / out_filename,
            X=X, y=y, meta=meta
        )
        print(f"Saved {X.shape[0]} windows to {out_filename}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

def main():
    # Process all trials for all participants in the dataset
    subjects = sorted(RAW_DIR.glob("SA*")) + sorted(RAW_DIR.glob("SE*"))
    for subject_dir in subjects:
        files = list(subject_dir.glob("*.txt"))
        for file_path in files:
            process_trial(file_path)

if __name__ == "__main__":
    main()