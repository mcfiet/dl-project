#!/usr/bin/env python
"""
Merge all CSV files in a directory into one CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CSVs in a folder into one file.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing CSV files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged.csv"),
        help="Output CSV path (default: merged.csv).",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for CSVs recursively.",
    )
    parser.add_argument(
        "--add-source",
        action="store_true",
        help="Add a 'source_file' column with the CSV filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    pattern = f"**/{args.pattern}" if args.recursive else args.pattern
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {pattern} in {input_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if args.add_source:
            df = df.copy()
            df["source_file"] = path.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged {len(files)} files -> {args.output} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
