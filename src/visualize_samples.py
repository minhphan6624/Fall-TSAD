from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parse_filename import parse_filename


RAW_ROOT = Path("data/raw/sisfall")
OUT_DIR = Path("figures/sisfall")
DEFAULT_FILES = (
    Path("SA01/D08_SA01_R01.txt"),
    Path("SA01/F08_SA01_R01.txt"),
)
FS_HZ = 200.0
ADXL345_RESOLUTION = 13
ADXL345_RANGE = 16  # +-16g
ACC_1_CONVERSION_FACTOR = (2 * ADXL345_RANGE) / (2**ADXL345_RESOLUTION)


def _resolve_input_files(raw_root: Path, provided_files: list[str]) -> list[Path]:
    if provided_files:
        files = [Path(file_path) for file_path in provided_files]
    else:
        files = [raw_root / rel_path for rel_path in DEFAULT_FILES]

    resolved_files: list[Path] = []
    for file_path in files:
        candidate = file_path
        if not candidate.is_absolute() and not str(candidate).startswith(str(raw_root)):
            candidate = raw_root / candidate

        if not candidate.exists():
            raise FileNotFoundError(f"Could not find input file: {candidate}")
        resolved_files.append(candidate)

    return resolved_files


def _plot_sample(
    file_path: Path,
    out_dir: Path,
    seconds: float,
    fs_hz: float,
) -> Path:
    meta = parse_filename(file_path.name)
    acc = _load_acc1_signal(file_path)

    n_initial = min(len(acc), int(seconds * fs_hz))
    if n_initial <= 0:
        raise ValueError("Initial window length must be > 0")

    smv = np.linalg.norm(acc, axis=1)

    t_all = np.arange(len(acc)) / fs_hz
    t_initial = np.arange(n_initial) / fs_hz

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t_initial, acc[:n_initial, 0], label="acc_x", linewidth=1.0)
    axes[0].plot(t_initial, acc[:n_initial, 1], label="acc_y", linewidth=1.0)
    axes[0].plot(t_initial, acc[:n_initial, 2], label="acc_z", linewidth=1.0)
    axes[0].set_title(f"Initial {seconds:.1f}s | {meta['activity']} {meta['subject']} R{meta['trial']:02d}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    # SMV
    axes[1].plot(t_all, smv, color="black", linewidth=1.0, label="SMV (raw)")
    axes[1].axvspan(0, seconds, color="tab:orange", alpha=0.2, label="Initial window")
    axes[1].set_title("Signal magnitude vector (full recording)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("SMV (g)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{file_path.stem}_overview.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def _load_acc1_signal(file_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            clean = line.strip().rstrip(";")
            if not clean:
                continue
            parts = [part.strip() for part in clean.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Invalid row at {file_path}:{line_no}")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not rows:
        raise ValueError(f"No samples found in {file_path}")

    acc_adc = np.asarray(rows, dtype=np.float32)
    return acc_adc * ACC_1_CONVERSION_FACTOR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one or two raw SISFall samples (acc1 + SMV)."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="Root folder containing SISFall text files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=(
            "Files to visualize. You can pass paths relative to --raw-root "
            "(e.g., SA01/D01_SA01_R01.txt). If omitted, one ADL and one fall example are used."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Number of initial seconds to show in the top panel.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=FS_HZ,
        help="Sampling frequency in Hz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Where figures are saved.",
    )

    args = parser.parse_args()
    input_files = _resolve_input_files(args.raw_root, args.files or [])

    print(f"Visualizing {len(input_files)} file(s)...")
    for file_path in input_files:
        output_path = _plot_sample(
            file_path=file_path,
            out_dir=args.out_dir,
            seconds=args.seconds,
            fs_hz=args.fs,
        )
        print(f"- {file_path} -> {output_path}")


if __name__ == "__main__":
    main()
