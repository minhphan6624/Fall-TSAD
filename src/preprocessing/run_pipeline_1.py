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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse raw datasets into the common interim trial pickle format."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("sisfall", "fallalld", "umafall", "upfall"),
        default=list("sisfall", "fallalld", "umafall", "upfall"),
        help="Datasets to parse. Defaults to all four datasets.",
    )
    
    args = parser.parsge_args()

    for dataset in args.datasets:
        print(f"\n=== Parsing {dataset} ===", flush=True)
        parser_module = importlib.import_module(PARSER_MODULES[dataset])
        parser_module.main()

    print("\nParsing completed successfully.")


if __name__ == "__main__":
    main()
