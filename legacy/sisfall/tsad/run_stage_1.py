from pathlib import Path
import re
import numpy as np
from sklearn.preprocessing import RobustScaler

from load_signal import load_signal
from segment_label import segment_and_label

RAW_DIR = Path("data/raw/sisfall/")
OUT_DIR = Path("data/processed/sisfall/tsad/windows")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filename pattern: D01_SA01_R01.txt
_NAME_RE = re.compile(r"^(?P<code>[DF]\d{2})_(?P<subject>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")

def parse_filename(name: str) -> dict:
    """ Parse a file and extract the important metadata """
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"Invalid filename: {name}")
    
    # Extract metadata from regex groups
    activity = m.group("code")
    subject = m.group("subject")    
    trial = m.group("trial")
    is_fall = activity.startswith("F")
    group = "young" if subject.startswith("SA") else "elderly"

    return {
        "activity": activity,
        "subject": subject,
        "group": group,
        "trial": trial,
        "is_fall": int(is_fall)
    }

from scipy.signal import butter, filtfilt
# ----- Low-pass filter ----- 
def butter_lowpass_filter(data, cutoff=5, fs=200, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def process_file(file_path):
    try:
        # Load metadata and signals (converted)
        meta = parse_filename(file_path.name)
        signals = load_signal(file_path)

        
        # Apply low-pass filter to data
        filtered = {
            name: butter_lowpass_filter(data)
            for name, data in signals.items()
        }

        # Calculate SMV (Signal Magnitude Vector)
        smv = np.sqrt(np.sum(filtered['acc1']**2, axis=1))

        combined_data = np.hstack([v for v in filtered.values()])

        # Segment & label
        X, y = segment_and_label(combined_data, smv, meta, window_size=200, stride=100)

        # Save windows + labels
        out_filename = f"{meta['activity']}_{meta['subject']}_{meta['trial']}.npz"
        np.savez_compressed(
            OUT_DIR / out_filename,
            X=X, y=y, meta=meta
        )
        print(f"Saved {X.shape[0]} windows to {out_filename}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

def main():
    subjects = sorted(RAW_DIR.glob("SA*")) + sorted(RAW_DIR.glob("SE*"))
    for subject_dir in subjects:
        files = list(subject_dir.glob("*.txt"))
        for file_path in files:
            process_file(file_path)

if __name__ == "__main__":
    main()