#!/usr/bin/env python3
"""
Build train/val/test splits by season.

Loads all CSVs from a directory (default: data/years/*.csv), infers the season
year (from a 'year' column or the filename), concatenates them, and writes
three CSVs with user-controlled year assignment, e.g.:

  python split_years_dataset.py \\
      --input-dir data/years \\
      --val-years 2024 \\
      --test-years 2025 \\
      --output-prefix data/grandprix_features

All remaining years not in val/test are placed into train.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split season CSVs into train/val/test by year.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/years"),
        help="Directory containing per-season CSVs (default: data/years)",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match season files (default: *.csv)",
    )
    p.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit train years. Remaining years (not in val/test) are also added to train.",
    )
    p.add_argument(
        "--val-years",
        type=int,
        nargs="+",
        default=[],
        help="Validation years.",
    )
    p.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=[],
        help="Test years.",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("data/grandprix_features"),
        help="Prefix for output CSVs (default: data/grandprix_features). _train/_val/_test will be appended.",
    )
    return p.parse_args()


def infer_year_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"(19|20)\\d{2}", path.stem)
    return int(m.group(0)) if m else None


def load_seasons(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {input_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if "year" not in df.columns:
            year = infer_year_from_filename(path)
            if year is None:
                raise ValueError(f"Cannot infer year for {path}; add a 'year' column or year in filename.")
            df["year"] = year
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def assign_years(
    all_years: Iterable[int],
    train_years: Optional[Iterable[int]],
    val_years: Iterable[int],
    test_years: Iterable[int],
) -> tuple[Set[int], Set[int], Set[int]]:
    all_years = set(int(y) for y in all_years)
    val_set = set(int(y) for y in val_years)
    test_set = set(int(y) for y in test_years)
    if val_set & test_set:
        raise ValueError(f"val/test overlap: {val_set & test_set}")

    train_set = set(int(y) for y in train_years) if train_years else set()
    if train_set & val_set or train_set & test_set:
        raise ValueError("train years overlap with val/test; adjust input.")

    remaining = all_years - val_set - test_set - train_set
    train_set |= remaining
    return train_set, val_set, test_set


def main() -> None:
    args = parse_args()

    df = load_seasons(args.input_dir, args.pattern)
    if "year" not in df.columns:
        raise ValueError("Joined dataframe has no 'year' column.")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    all_years = sorted(df["year"].unique())
    train_years, val_years, test_years = assign_years(
        all_years=all_years,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
    )

    train_df = df[df["year"].isin(train_years)].copy()
    val_df = df[df["year"].isin(val_years)].copy() if val_years else pd.DataFrame()
    test_df = df[df["year"].isin(test_years)].copy() if test_years else pd.DataFrame()

    prefix = args.output_prefix
    out_train = prefix.with_name(prefix.name + "_train.csv")
    out_val = prefix.with_name(prefix.name + "_val.csv")
    out_test = prefix.with_name(prefix.name + "_test.csv")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_train, index=False)
    if not val_df.empty:
        val_df.to_csv(out_val, index=False)
    else:
        out_val.write_text("")
    if not test_df.empty:
        test_df.to_csv(out_test, index=False)
    else:
        out_test.write_text("")

    print(f"Years found: {all_years}")
    print(f"Train years: {sorted(train_years)} -> {len(train_df)} rows -> {out_train}")
    print(f"Val years:   {sorted(val_years)} -> {len(val_df)} rows -> {out_val}")
    print(f"Test years:  {sorted(test_years)} -> {len(test_df)} rows -> {out_test}")


if __name__ == "__main__":
    main()
