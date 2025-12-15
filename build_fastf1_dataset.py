#!/usr/bin/env python
"""
Build a classification-ready dataset from FastF1 data.

Features include:
 - driver, constructor, and circuit identifiers
 - grid position and qualifying deltas (overall + teammate)
 - cumulative season points for driver and constructor
 - rolling form (last 3 race points average)
 - simple context flags (street circuit, wet race)
Label:
 - whether the driver scored points in the race
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

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


def safe_load_session(
    year: int,
    round_number: int,
    code: str,
    *,
    load_weather: bool = False,
) -> Optional[fastf1.core.Session]:
    try:
        session = fastf1.get_session(year, round_number, code)
        session.load(telemetry=False, weather=load_weather)
        return session
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to load {year} round {round_number} session {code}: {exc}")
        return None


def normalize_identifier(value: Optional[str]) -> str:
    """Normalize IDs to lowercase snake_case, fallback to 'unknown'."""
    if value is None:
        return "unknown"
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return cleaned or "unknown"


def build_street_circuit_lookup() -> Set[str]:
    keywords = {
        "monaco",
        "singapore",
        "baku",
        "jeddah",
        "miami",
        "las_vegas",
        "vegas",
        "montreal",
        "melbourne",
        "saudi",
        "azerbaijan",
    }
    return keywords


def infer_wet_race(session: fastf1.core.Session) -> int:
    """Return 1 if rainfall was recorded during the session, else 0."""
    try:
        weather = session.weather_data
    except DataNotLoadedError:
        return 0

    if weather is None or weather.empty:
        return 0

    if "Rainfall" in weather.columns:
        return int(bool(weather["Rainfall"].fillna(False).any()))
    return 0


def best_quali_lap_times(quali: fastf1.core.Session) -> Dict[str, float]:
    """Compute best qualifying lap time (seconds) per driver abbreviation."""
    times: Dict[str, float] = {}
    try:
        laps = quali.laps
    except DataNotLoadedError:
        laps = None

    if laps is not None and not laps.empty:
        for driver_number in quali.drivers or []:
            info = quali.get_driver(driver_number)
            abb = info.get("Abbreviation", str(driver_number))
            driver_laps = laps.pick_drivers(abb)
            if driver_laps.empty:
                driver_laps = laps.pick_drivers(driver_number)
            if driver_laps.empty:
                continue
            best = driver_laps["LapTime"].dt.total_seconds().min()
            if pd.notna(best):
                times[abb] = float(best)

    if not times and quali.results is not None and not quali.results.empty:
        # Fallback to Q1/Q2/Q3 columns when lap data is missing.
        for _, row in quali.results.iterrows():
            fastest = None
            for col in ("Q1", "Q2", "Q3"):
                if col in row and not pd.isna(row[col]):
                    val = pd.to_timedelta(row[col]).total_seconds()
                    fastest = val if fastest is None else min(fastest, val)
            if fastest is not None:
                key = row.get("Abbreviation") or row.get("DriverNumber")
                times[str(key)] = float(fastest)

    return times


def collect_rows(
    years: Iterable[int],
    session_type: str,
    limit_rounds: Optional[int],
) -> List[Dict]:
    driver_points: Dict[str, float] = defaultdict(float)
    driver_recent_points: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
    team_points: Dict[str, float] = defaultdict(float)
    street_keywords = build_street_circuit_lookup()
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
            quali = safe_load_session(year, round_number, "Q", load_weather=False)
            race = safe_load_session(year, round_number, session_type, load_weather=True)
            if quali is None or race is None:
                continue

            if race.drivers is None or len(race.drivers) == 0:
                print(
                    f"[warn] No drivers found for {year} round {round_number} "
                    f"{event['EventName']}; skipping event."
                )
                continue

            race_results = race.results
            quali_results = quali.results
            best_quali_times = best_quali_lap_times(quali)
            session_best_quali = min(best_quali_times.values()) if best_quali_times else None

            event_name = str(event.get("EventName", ""))
            location = str(event.get("Location", event_name))
            circuit_id = normalize_identifier(location or event_name)
            street_flag = 1 if any(k in f"{location} {event_name}".lower() for k in street_keywords) else 0
            wet_flag = infer_wet_race(race)
            team_to_drivers: Dict[str, List[str]] = defaultdict(list)
            driver_info_by_number: Dict[str, Dict] = {}

            for driver_number in race.drivers:
                info = race.get_driver(driver_number)
                driver_info_by_number[driver_number] = info
                abb = info.get("Abbreviation", str(driver_number))
                team = info.get("TeamName", info.get("Team"))
                if team:
                    team_to_drivers[team].append(abb)

            for driver_number in race.drivers:
                info = driver_info_by_number[driver_number]
                abb = info.get("Abbreviation", str(driver_number))
                team = info.get("TeamName", info.get("Team"))

                quali_pos = None
                if quali_results is not None and not quali_results.empty:
                    qrow = quali_results[quali_results["DriverNumber"] == driver_number]
                    if qrow.empty:
                        qrow = quali_results[quali_results["Abbreviation"] == abb]
                    if not qrow.empty:
                        pos_value = qrow["Position"].iloc[0]
                        # Some events have missing/NaN positions; keep as None instead of raising.
                        quali_pos = None if pd.isna(pos_value) else int(pos_value)

                points_awarded = 0.0
                if race_results is not None and not race_results.empty:
                    rrow = race_results[race_results["DriverNumber"] == driver_number]
                    if rrow.empty:
                        rrow = race_results[race_results["Abbreviation"] == abb]
                    if not rrow.empty:
                        points_awarded = float(rrow["Points"].iloc[0])

                prev_points = driver_points[abb]
                team_prev_points = team_points[team] if team else 0.0
                rolling_pts = driver_recent_points[abb]
                rolling_avg_last3 = (sum(rolling_pts) / len(rolling_pts)) if rolling_pts else 0.0

                driver_best_quali = best_quali_times.get(abb)
                quali_delta = (
                    driver_best_quali - session_best_quali
                    if driver_best_quali is not None and session_best_quali is not None
                    else None
                )
                teammate_delta = None
                if team and driver_best_quali is not None:
                    teammate_times = [
                        best_quali_times.get(other)
                        for other in team_to_drivers.get(team, [])
                        if other != abb
                    ]
                    teammate_times = [t for t in teammate_times if t is not None]
                    if teammate_times:
                        teammate_delta = driver_best_quali - min(teammate_times)

                driver_id = normalize_identifier(
                    info.get("LastName")
                    or info.get("FamilyName")
                    or info.get("FullName")
                    or abb
                )
                constructor_id = normalize_identifier(team) if team else "unknown"

                rows.append(
                    {
                        "driver_id": driver_id,
                        "constructor_id": constructor_id,
                        "circuit_id": circuit_id,
                        "grid_position": quali_pos,
                        "quali_delta": quali_delta,
                        "quali_tm_delta": teammate_delta,
                        "season_pts_driver": prev_points,
                        "season_pts_team": team_prev_points,
                        "last_3_avg": rolling_avg_last3,
                        "is_street_circuit": street_flag,
                        "is_wet": wet_flag,
                        "points_scored": 1 if points_awarded > 0 else 0,
                    }
                )

                driver_points[abb] = prev_points + points_awarded
                driver_recent_points[abb].append(points_awarded)
                if team:
                    team_points[team] = team_prev_points + points_awarded

    return rows


def build_dataset(
    years: Iterable[int],
    session_type: str,
    limit_rounds: Optional[int],
) -> pd.DataFrame:
    rows = collect_rows(
        years=years,
        session_type=session_type,
        limit_rounds=limit_rounds,
    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "driver_id",
                "constructor_id",
                "circuit_id",
                "grid_position",
                "quali_delta",
                "quali_tm_delta",
                "season_pts_driver",
                "season_pts_team",
                "last_3_avg",
                "is_street_circuit",
                "is_wet",
                "points_scored",
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    enable_cache(args.cache_dir)

    df = build_dataset(args.years, args.session_type, args.limit_rounds)
    if df.empty:
        print("No data collected; check the chosen years/rounds.")
        return

    feature_cols = [
        "driver_id",
        "constructor_id",
        "circuit_id",
        "grid_position",
        "quali_delta",
        "quali_tm_delta",
        "season_pts_driver",
        "season_pts_team",
        "last_3_avg",
        "is_street_circuit",
        "is_wet",
    ]
    target_col = "points_scored"
    print(
        "Built dataset with "
        f"{len(df)} rows. Feature columns: {feature_cols}. Target: {target_col}."
    )

    df["driver_id"] = df["driver_id"].fillna("unknown")
    df["constructor_id"] = df["constructor_id"].fillna("unknown")
    df["circuit_id"] = df["circuit_id"].fillna("unknown")
    df["grid_position"] = df["grid_position"].fillna(-1).astype(int)
    mean_quali_delta = df["quali_delta"].mean()
    df["quali_delta"] = df["quali_delta"].fillna(0.0 if pd.isna(mean_quali_delta) else mean_quali_delta)
    df["quali_tm_delta"] = df["quali_tm_delta"].fillna(0.0)
    df["season_pts_driver"] = df["season_pts_driver"].fillna(0.0)
    df["season_pts_team"] = df["season_pts_team"].fillna(0.0)
    df["last_3_avg"] = df["last_3_avg"].fillna(0.0)
    df["is_street_circuit"] = df["is_street_circuit"].fillna(0).astype(int)
    df["is_wet"] = df["is_wet"].fillna(0).astype(int)
    df["points_scored"] = df["points_scored"].fillna(0).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
