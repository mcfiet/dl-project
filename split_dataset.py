#!/usr/bin/env python3
"""
Split a CSV into train/val/test with stratification on the target.
Usage:
  python split_dataset.py --data data/grandprix_features_2.csv --target scored_points \
      --test-size 0.2 --val-size 0.25 --random-state 42
This yields 60/20/20 by default (val-size is a fraction of the remaining train).
Outputs three CSVs alongside the input (train/val/test suffixes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a CSV into train/val/test sets.")
    p.add_argument("--data", type=Path, required=True, help="Path to input CSV")
    p.add_argument("--target", type=str, required=True, help="Target column for stratification")
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

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df[args.target], random_state=args.random_state
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=args.val_size,
        stratify=train_df[args.target],
        random_state=args.random_state,
    )

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
