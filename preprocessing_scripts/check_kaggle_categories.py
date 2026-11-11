"""
check_kaggle_categories.py

List unique values in the Category column of the raw Kaggle AI-tools file.
Read-only: it never modifies data/raw/kaggle.csv.

Usage (from the project root):
    python preprocessing_scripts/check_kaggle_categories.py
    python preprocessing_scripts/check_kaggle_categories.py --path data/raw/kaggle.csv
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "raw" / "kaggle.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to the raw Kaggle CSV (default: data/raw/kaggle.csv)",
    )
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Error: file not found: {args.path}")

    df = pd.read_csv(args.path, dtype=str, keep_default_na=False)
    if "Category" not in df.columns:
        raise SystemExit(
            "Error: missing 'Category' column. Found: "
            + ", ".join(df.columns.astype(str))
        )

    series = df["Category"]
    blank = (series.str.strip() == "").sum()
    unique = sorted({v.strip() for v in series if v.strip()})

    print(f"File: {args.path}")
    print(f"Rows: {len(df)}")
    print(f"Blank/empty Category values: {blank}")
    print(f"Unique Category values: {len(unique)}")
    print()
    for category in unique:
        count = (series.str.strip() == category).sum()
        print(f"  {count:6d}  {category}")


if __name__ == "__main__":
    main()
