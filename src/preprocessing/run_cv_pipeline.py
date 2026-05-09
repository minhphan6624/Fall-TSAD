import argparse

from src.preprocessing.build_splits import DEFAULT_N_FOLDS, DEFAULT_SEED, KFOLD_SPLIT_PROTOCOL
from src.preprocessing.run_pipeline_2 import INTERIM_PICKLE_NAMES, run_pipeline
from src.preprocessing.window_trials import DEFAULT_OVERLAP, DEFAULT_WINDOW_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate processed artifacts for all subject-wise CV folds."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(INTERIM_PICKLE_NAMES.keys()),
        help="Dataset name to process using the repo's standard directory layout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used by the subject fold builder.",
    )
    parser.add_argument(
        "--protocol",
        choices=(KFOLD_SPLIT_PROTOCOL,),
        default=KFOLD_SPLIT_PROTOCOL,
        help="Cross-validation split protocol.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of subject folds to generate.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help="Fractional window overlap in [0, 1).",
    )
    parser.add_argument(
        "--target-sampling-rate-hz",
        type=float,
        default=None,
        help="Optional target sampling rate. If provided, trial acc arrays are resampled before windowing.",
    )
    parser.add_argument(
        "--allow-upsample",
        action="store_true",
        help="Allow upsampling when the target sampling rate is higher than a trial's source rate.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Processed dataset prefix. Fold outputs are named <output-prefix>_fold<index>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for fold_index in range(args.n_folds):
        output_dataset = f"{args.output_prefix}_fold{fold_index}"
        out_dir = run_pipeline(
            dataset=args.dataset,
            seed=args.seed,
            split_protocol=args.protocol,
            n_folds=args.n_folds,
            fold_index=fold_index,
            window_seconds=args.window_seconds,
            overlap=args.overlap,
            target_sampling_rate_hz=args.target_sampling_rate_hz,
            allow_upsample=args.allow_upsample,
            output_dataset=output_dataset,
        )
        print(f"Saved fold {fold_index} processed artifacts to {out_dir}")


if __name__ == "__main__":
    main()
