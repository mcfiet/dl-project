#!/usr/bin/env python3
"""
Split a CSV into train/val/test **by year**, keeping temporal order.
Usage:
  python split_dataset.py --data data/grandprix_features.csv --target points_scored \
      --year-column year --test-size 0.2 --val-size 0.25
This yields oldest years for train, next block for val, newest for test
(val-size is a fraction of the remaining non-test years).
Outputs three CSVs alongside the input (train/val/test suffixes).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a CSV into train/val/test sets.")
    p.add_argument("--data", type=Path, required=True, help="Path to input CSV")
    p.add_argument("--target", type=str, required=True, help="Target column for stratification")
    p.add_argument(
        "--year-column",
        type=str,
        default="year",
        help="Column containing season/year used for temporal splits (default: year)",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default 0.2)")
    p.add_argument(
        "--val-size",
        type=float,
        default=0.25,
        help="Validation fraction of the remaining train (default 0.25 => 60/20/20 total)",
    )
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--output-prefix", type=str, default=None, help="Optional prefix for output files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.data}")
    if args.year_column not in df.columns:
        raise ValueError(f"Year column '{args.year_column}' not found in {args.data}")

    # ensure year is numeric and drop rows without year
    if not pd.api.types.is_numeric_dtype(df[args.year_column]):
        try:
            df[args.year_column] = df[args.year_column].astype(int)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Year column '{args.year_column}' is not numeric: {exc}") from exc
    df = df.dropna(subset=[args.year_column])
    years = sorted(df[args.year_column].unique())
    if len(years) < 3:
        raise ValueError(f"Need at least 3 distinct years to split; found {years}")

    n_years = len(years)
    n_test_years = max(1, math.ceil(n_years * args.test_size)) if args.test_size > 0 else 0
    test_years = years[-n_test_years:] if n_test_years else []
    train_val_years = years[:-n_test_years] if n_test_years else years
    if not train_val_years:
        raise ValueError("No years left for train/val after allocating test split.")

    n_val_years = (
        max(1, math.ceil(len(train_val_years) * args.val_size)) if args.val_size > 0 else 0
    )
    val_years = train_val_years[-n_val_years:] if n_val_years else []
    train_years = train_val_years[:-n_val_years] if n_val_years else train_val_years

    if not train_years:
        raise ValueError("No years allocated to train split; adjust val/test sizes.")

    train_df = df[df[args.year_column].isin(train_years)].copy()
    val_df = df[df[args.year_column].isin(val_years)].copy() if val_years else pd.DataFrame()
    test_df = df[df[args.year_column].isin(test_years)].copy() if test_years else pd.DataFrame()

    print(f"Years -> train: {train_years} | val: {val_years} | test: {test_years}")
    if val_df.empty or test_df.empty:
        print("[warn] Val or test split is empty; check chosen sizes/years.")

    prefix = args.output_prefix or args.data.with_suffix("")
    prefix = Path(prefix)
    out_train = prefix.with_name(prefix.name + "_train.csv")
    out_val = prefix.with_name(prefix.name + "_val.csv")
    out_test = prefix.with_name(prefix.name + "_test.csv")

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"Saved train: {out_train} ({len(train_df)} rows)")
    print(f"Saved val:   {out_val} ({len(val_df)} rows)")
    print(f"Saved test:  {out_test} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
