#!/usr/bin/env python3
"""
Compute the average retirement rate across given F1 seasons using FastF1.

The script queries FastF1 directly (no local datasets) and reports the overall
percentage of retirements across all drivers and races in the provided years.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import fastf1
from fastf1.core import Session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate average retirement rate for given F1 seasons using FastF1."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Season years to include (e.g. --years 2019 2020 2021).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fastf1_cache"),
        help="Directory for FastF1 cache (reused between runs).",
    )
    return parser.parse_args()


def quiet_logs() -> None:
    """Silence verbose FastF1/requests logging."""
    for name in ["fastf1", "fastf1.logger", "fastf1._api", "fastf1.req", "urllib3"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.basicConfig(level=logging.ERROR)


def enable_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir, ignore_version=True)


def is_retired(status: str) -> bool:
    """
    Heuristic for race retirement:
    - Consider finished if status starts with "Finished", "+" (laps behind) or "Lap".
    - Everything else (accident, mechanical, disqualified, not classified, etc.) counts as retirement.
    """
    cleaned = (status or "").strip().lower()
    return not (
        cleaned.startswith("finished")
        or cleaned.startswith("+")
        or cleaned.startswith("lap")
        or cleaned.startswith("laps")
    )


def count_retirements_for_year(year: int) -> Tuple[int, int]:
    """Return (retirements, total_classified) for the given season."""
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to load schedule for %s: %s", year, exc)
        return 0, 0

    # Only races with a valid round number.
    races = schedule[schedule["RoundNumber"] > 0]
    retires = 0
    total = 0

    for _, event in races.iterrows():
        round_number = int(event["RoundNumber"])
        session: Session = fastf1.get_session(year, round_number, "R")
        try:
            session.load(telemetry=False, weather=False)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to load %s %s: %s", year, round_number, exc)
            continue

        if session.results is None or session.results.empty:
            logging.warning("No results for %s %s", year, round_number)
            continue

        for _, row in session.results.iterrows():
            status = str(row.get("Status", "") or "")
            total += 1
            if is_retired(status):
                retires += 1

    return retires, total


def summarize(years: Iterable[int]) -> None:
    per_year = []
    total_retires = 0
    total_entries = 0

    for year in years:
        retires, entries = count_retirements_for_year(year)
        if entries == 0:
            print(f"{year}: keine Daten (0 Einträge)")
            continue
        rate = (retires / entries) * 100
        per_year.append((year, rate))
        total_retires += retires
        total_entries += entries
        print(f"{year}: {rate:.2f}% Ausfallquote ({retires}/{entries})")

    if total_entries == 0:
        print("Keine Daten gefunden.")
        return

    overall = (total_retires / total_entries) * 100
    print(f"Gesamt: {overall:.2f}% Ausfallquote über {total_entries} Starts")


def main() -> None:
    args = parse_args()
    quiet_logs()
    enable_cache(args.cache_dir)
    summarize(args.years)


if __name__ == "__main__":
    main()
