from pathlib import Path
import numpy as np

from load_signal import load_signal
from parse_filename import parse_filename
from filter import butter_lowpass_filter

RAW_DIR = Path("data/raw/sisfall/")
OUT_DIR = Path("data/processed/sisfall/stage_1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_stage_1():
    subjects = sorted(RAW_DIR.glob("SA*")) + sorted(RAW_DIR.glob("SE*"))
    for subject_dir in subjects:
        files = list(subject_dir.glob("*.txt"))
        for file_path in files:
            try:
                metadata = parse_filename(file_path.name)
                signals = load_signal(file_path)

                filtered = {
                    name: butter_lowpass_filter(sig, cutoff=5, fs=200)
                    for name, sig in signals.items()
                }
                
                # Save the filtered signals
                filename, ext = file_path.name.split('.', 1)
                out_file = OUT_DIR / f"{filename}.npz"
                np.savez_compressed(out_file, **filtered, metadata=metadata)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    run_stage_1()