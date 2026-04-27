from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# The dataset parsers are also executable as scripts and import their sibling
# common.py module directly. Add this directory so this runner works both as:
#   python3 src/preprocessing/parse_all.py
# and:
#   python3 -m src.preprocessing.parse_all
PREPROCESSING_DIR = Path(__file__).resolve().parent
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

PARSER_MODULES = {
    "sisfall": "parse_sisfall",
    "fallalld": "parse_fallalld",
    "umafall": "parse_umafall",
    "upfall": "parse_upfall",
}

DEFAULT_DATASET_ORDER = ("sisfall", "fallalld", "umafall", "upfall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse raw datasets into the common interim trial pickle format."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASET_ORDER,
        default=list(DEFAULT_DATASET_ORDER),
        help="Datasets to parse. Defaults to all four datasets.",
    )
    return parser.parse_args()


def run_parsers(datasets: list[str]) -> None:
    for dataset in datasets:
        print(f"\n=== Parsing {dataset} ===", flush=True)
        parser_module = importlib.import_module(PARSER_MODULES[dataset])
        parser_module.main()


def main() -> None:
    args = parse_args()
    run_parsers(datasets=args.datasets)
    print("\nParsing completed successfully.")


if __name__ == "__main__":
    main()
