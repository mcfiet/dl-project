#!/usr/bin/env python
"""
Build a classification-ready dataset from FastF1 data.

Features include:
 - average race lap time per driver
 - qualifying position
 - cumulative points scored before the event
Label:
 - whether the driver scored points in the race
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import fastf1
from fastf1.core import DataNotLoadedError
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FastF1 data and build features for a classification model."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Season years to include. Example: --years 2022 2023 2024",
    )
    parser.add_argument(
        "--session-type",
        default="R",
        choices=["R", "S"],
        help="Race session to use for lap data: R = race, S = sprint (default: R).",
    )
    parser.add_argument(
        "--limit-rounds",
        type=int,
        default=None,
        help="Optional: only process the first N rounds of each year (useful for quick tests).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/fastf1_cache"),
        help="Directory to store FastF1 cache (reused between runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/grandprix_features.csv"),
        help="Where to write the resulting CSV.",
    )
    return parser.parse_args()


def enable_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir, ignore_version=True)


def safe_load_session(year: int, round_number: int, code: str) -> Optional[fastf1.core.Session]:
    try:
        session = fastf1.get_session(year, round_number, code)
        session.load(telemetry=False, weather=False)
        return session
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to load {year} round {round_number} session {code}: {exc}")
        return None


def collect_rows(
    years: Iterable[int],
    session_type: str,
    limit_rounds: Optional[int],
) -> List[Dict]:
    driver_points: Dict[str, float] = defaultdict(float)
    rows: List[Dict] = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to load schedule for {year}: {exc}; skipping year.")
            continue
        race_schedule = schedule[schedule["RoundNumber"] > 0]
        if limit_rounds:
            race_schedule = race_schedule.head(limit_rounds)

        iterator = tqdm(
            race_schedule.iterrows(),
            total=len(race_schedule),
            desc=f"{year} season",
            disable=len(race_schedule) == 0,
        )

        for _, event in iterator:
            round_number = int(event["RoundNumber"])
            quali = safe_load_session(year, round_number, "Q")
            race = safe_load_session(year, round_number, session_type)
            if quali is None or race is None:
                continue

            try:
                race_laps = race.laps.pick_quicklaps()
            except DataNotLoadedError:
                print(
                    f"[warn] Laps not loaded for {year} round {round_number} "
                    f"{event['EventName']}; skipping event."
                )
                continue

            if race.drivers is None or len(race.drivers) == 0:
                print(
                    f"[warn] No drivers found for {year} round {round_number} "
                    f"{event['EventName']}; skipping event."
                )
                continue

            race_results = race.results
            quali_results = quali.results

            for driver_number in race.drivers:
                info = race.get_driver(driver_number)
                abb = info.get("Abbreviation", str(driver_number))
                team = info.get("TeamName", info.get("Team"))

                driver_laps = race_laps.pick_drivers(abb)
                avg_lap = (
                    driver_laps["LapTime"].dt.total_seconds().mean()
                    if not driver_laps.empty
                    else float("nan")
                )

                quali_pos = None
                if quali_results is not None and not quali_results.empty:
                    qrow = quali_results[quali_results["DriverNumber"] == driver_number]
                    if qrow.empty:
                        qrow = quali_results[quali_results["Abbreviation"] == abb]
                    if not qrow.empty:
                        pos_value = qrow["Position"].iloc[0]
                        # Some events have missing/NaN positions; keep as None instead of raising.
                        quali_pos = None if pd.isna(pos_value) else int(pos_value)

                finish_pos = None
                points_awarded = 0.0
                if race_results is not None and not race_results.empty:
                    rrow = race_results[race_results["DriverNumber"] == driver_number]
                    if rrow.empty:
                        rrow = race_results[race_results["Abbreviation"] == abb]
                    if not rrow.empty:
                        raw_pos = rrow["Position"].iloc[0]
                        finish_pos = None if pd.isna(raw_pos) else int(raw_pos)
                        points_awarded = float(rrow["Points"].iloc[0])

                prev_points = driver_points[abb]

                rows.append(
                    {
                        "year": year,
                        "round": round_number,
                        "event": str(event["EventName"]),
                        "driver": abb,
                        "team": team,
                        "quali_position": quali_pos,
                        "avg_race_lap_time_s": avg_lap,
                        "finish_position": finish_pos,
                        "points_awarded": points_awarded,
                        "prev_points_total": prev_points,
                        "scored_points": 1 if points_awarded > 0 else 0,
                    }
                )

                driver_points[abb] = prev_points + points_awarded

    return rows


def main() -> None:
    args = parse_args()
    enable_cache(args.cache_dir)

    rows = collect_rows(
        years=args.years,
        session_type=args.session_type,
        limit_rounds=args.limit_rounds,
    )

    if not rows:
        print("No data collected; check the chosen years/rounds.")
        return

    df = pd.DataFrame(rows)

    feature_cols = ["avg_race_lap_time_s", "quali_position", "prev_points_total"]
    target_col = "scored_points"
    print(
        "Built dataset with "
        f"{len(df)} rows. Feature columns: {feature_cols}. Target: {target_col}"
    )

    # Fill obvious gaps to avoid NaNs during training
    df["avg_race_lap_time_s"] = df["avg_race_lap_time_s"].fillna(df["avg_race_lap_time_s"].mean())
    df["quali_position"] = df["quali_position"].fillna(-1)
    df["prev_points_total"] = df["prev_points_total"].fillna(0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
