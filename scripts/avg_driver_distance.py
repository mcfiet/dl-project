#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute average adjacent distance between drivers per race for a target column."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with race rows (one row per driver).",
    )
    parser.add_argument(
        "--target",
        default="gap_to_winner",
        help="Target column to compare between drivers.",
    )
    parser.add_argument(
        "--group-cols",
        default="year,round_number",
        help="Comma-separated columns that identify a race.",
    )
    parser.add_argument(
        "--mae",
        type=float,
        default=None,
        help="Optional MAE value to compare against the average distance.",
    )
    return parser.parse_args()


def avg_adjacent_abs(values):
    if len(values) < 2:
        return None
    values = sorted(values)
    total = 0.0
    for i in range(1, len(values)):
        total += abs(values[i] - values[i - 1])
    return total / (len(values) - 1)


def main():
    args = parse_args()
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    groups = defaultdict(list)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row.get(col, "").strip() for col in group_cols)
            raw = row.get(args.target, "").strip()
            if raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            groups[key].append(value)

    per_race = []
    total_adjacent = 0.0
    total_count = 0
    for key, values in groups.items():
        dist = avg_adjacent_abs(values)
        if dist is not None:
            per_race.append(dist)
            total_adjacent += dist * (len(values) - 1)
            total_count += len(values) - 1

    if not per_race:
        raise SystemExit("No valid races with at least two drivers found.")

    avg_of_races = mean(per_race)
    overall_avg = total_adjacent / total_count if total_count else 0.0
    print(f"Races evaluated: {len(per_race)}")
    print(f"Average adjacent distance per race (mean of races): {avg_of_races:.4f}")
    print(f"Overall average adjacent distance (all gaps pooled): {overall_avg:.4f}")

    if args.mae is not None:
        ratio = args.mae / overall_avg if overall_avg else float("inf")
        print(f"MAE: {args.mae:.4f}")
        print(f"MAE / avg_distance: {ratio:.4f}")


if __name__ == "__main__":
    main()
