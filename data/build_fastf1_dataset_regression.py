#!/usr/bin/env python
"""
Build a regression-ready dataset from FastF1 data.

Features include:
 - driver, constructor, and circuit identifiers
 - grid position and qualifying deltas (overall + teammate)
 - cumulative season points for driver and constructor
- rolling form (last 3 race points average)
 - simple context flags (street circuit, wet race)
Target:
 - konsistente Racezeit pro Fahrer in Sekunden:
     * Sieger: offizielle Rennzeit
     * Alle anderen: Siegerzeit + gemeldeter Gap (falls vorhanden)
     * DNFs/fehlende Zeit -> NaN, Status wird mit ausgegeben
 - leichte Ausreißer-Glättung über Quantil-Clipping
Weitere abgeleitete Targets:
 - gap_to_winner (Sekunden, Sieger=0)
 - race_time_per_lap (Sekunden pro Runde, falls Laps bekannt)
Zusätzliche Trainingsfeatures:
 - Trainingssessions FP1/FP2/FP3: beste Rundenzeit (s), Rundenanzahl, Relativzeit zum Session-Best
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import fastf1
from fastf1.core import DataNotLoadedError
import pandas as pd
from tqdm import tqdm
import logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FastF1 data and build features for a race-time regression model."
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


def quiet_logs() -> None:
    """Silence verbose FastF1/requests logging."""
    for name in ["fastf1", "fastf1.logger", "fastf1._api", "fastf1.req", "urllib3"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.basicConfig(level=logging.ERROR)


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
    except Exception:  # noqa: BLE001
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


def parse_time_to_seconds(value) -> Optional[float]:
    """Convert FastF1 result time/gap to seconds."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    try:
        td = pd.to_timedelta(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(td):
        return None
    return float(td.total_seconds())


def is_classified(status: str) -> bool:
    """Return True if driver finished/classified; False for DNFs/DSQ."""
    cleaned = (status or "").strip().lower()
    return cleaned.startswith("finished") or cleaned.startswith("+") or cleaned.startswith("lap")


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


def session_best_laps(session: Optional[fastf1.core.Session]) -> tuple[Dict[str, float], Dict[str, int], Optional[float]]:
    """
    Return per-driver best lap (seconds), lap counts, and session-best lap.
    """
    times: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    session_best: Optional[float] = None
    if session is None:
        return times, counts, session_best
    try:
        laps = session.laps
    except DataNotLoadedError:
        return times, counts, session_best
    if laps is None or laps.empty:
        return times, counts, session_best
    for driver_number in session.drivers or []:
        info = session.get_driver(driver_number)
        abb = info.get("Abbreviation", str(driver_number))
        driver_laps = laps.pick_drivers(abb)
        if driver_laps.empty:
            driver_laps = laps.pick_drivers(driver_number)
        if driver_laps.empty:
            continue
        counts[abb] = len(driver_laps)
        best = driver_laps["LapTime"].dt.total_seconds().min()
        if pd.notna(best):
            times[abb] = float(best)
            session_best = best if session_best is None else min(session_best, best)
    return times, counts, session_best


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
        except Exception as exc:
            continue
        race_schedule = schedule[schedule["RoundNumber"] > 0]
        if limit_rounds:
            race_schedule = race_schedule.head(limit_rounds)
        total_events = len(race_schedule)
        loaded_events = 0

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
            fp1 = safe_load_session(year, round_number, "FP1", load_weather=False)
            fp2 = safe_load_session(year, round_number, "FP2", load_weather=False)
            fp3 = safe_load_session(year, round_number, "FP3", load_weather=False)
            if quali is None or race is None:
                continue

            if race.drivers is None or len(race.drivers) == 0:
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
            winner_time_seconds: Optional[float] = None

            fp1_times, fp1_counts, fp1_best_session = session_best_laps(fp1)
            fp2_times, fp2_counts, fp2_best_session = session_best_laps(fp2)
            fp3_times, fp3_counts, fp3_best_session = session_best_laps(fp3)

            if race_results is not None and not race_results.empty:
                winner_rows = race_results[race_results["Position"] == 1]
                if not winner_rows.empty:
                    w_time = winner_rows["Time"].iloc[0] if "Time" in winner_rows.columns else None
                    winner_time_seconds = parse_time_to_seconds(w_time)

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
                        quali_pos = None if pd.isna(pos_value) else int(pos_value)

                points_awarded = 0.0
                race_time_seconds: Optional[float] = None
                status_val = ""
                pos_val: Optional[int] = None
                laps_val: Optional[int] = None
                if race_results is not None and not race_results.empty:
                    rrow = race_results[race_results["DriverNumber"] == driver_number]
                    if rrow.empty:
                        rrow = race_results[race_results["Abbreviation"] == abb]
                    if not rrow.empty:
                        points_awarded = float(rrow["Points"].iloc[0])
                        status_val = str(rrow["Status"].iloc[0]) if "Status" in rrow.columns else ""
                        pos_raw = rrow["Position"].iloc[0] if "Position" in rrow.columns else None
                        if pd.notna(pos_raw):
                            pos_val = int(pos_raw)
                        if "Laps" in rrow.columns:
                            laps_raw = rrow["Laps"].iloc[0]
                            if pd.notna(laps_raw):
                                try:
                                    laps_val = int(laps_raw)
                                except Exception:
                                    laps_val = None
                        if is_classified(status_val):
                            if "Time" in rrow.columns:
                                time_value = rrow["Time"].iloc[0]
                                parsed = parse_time_to_seconds(time_value)
                                # Sieger: direkte Zeit
                                if pos_val == 1:
                                    race_time_seconds = parsed
                                    if winner_time_seconds is None:
                                        winner_time_seconds = parsed
                                else:
                                    if winner_time_seconds is not None and parsed is not None:
                                        race_time_seconds = winner_time_seconds + parsed
                                    else:
                                        # Fallback, falls keine Siegerzeit verfügbar ist
                                        race_time_seconds = parsed

                gap_to_winner = None
                if pos_val == 1 and race_time_seconds is not None:
                    gap_to_winner = 0.0
                elif winner_time_seconds is not None and race_time_seconds is not None:
                    gap_to_winner = race_time_seconds - winner_time_seconds

                race_time_per_lap = None
                if race_time_seconds is not None and laps_val and laps_val > 0:
                    race_time_per_lap = race_time_seconds / laps_val

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

                fp1_best = fp1_times.get(abb)
                fp2_best = fp2_times.get(abb)
                fp3_best = fp3_times.get(abb)
                fp1_rel = fp1_best - fp1_best_session if fp1_best is not None and fp1_best_session else None
                fp2_rel = fp2_best - fp2_best_session if fp2_best is not None and fp2_best_session else None
                fp3_rel = fp3_best - fp3_best_session if fp3_best is not None and fp3_best_session else None
                fp1_laps = fp1_counts.get(abb)
                fp2_laps = fp2_counts.get(abb)
                fp3_laps = fp3_counts.get(abb)

                rows.append(
                    {
                        "year": year,
                        "round_number": round_number,
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
                        "race_time": race_time_seconds,
                        "race_status": status_val or "",
                        "gap_to_winner": gap_to_winner,
                        "laps": laps_val,
                        "race_time_per_lap": race_time_per_lap,
                        "fp1_best": fp1_best,
                        "fp2_best": fp2_best,
                        "fp3_best": fp3_best,
                        "fp1_rel": fp1_rel,
                        "fp2_rel": fp2_rel,
                        "fp3_rel": fp3_rel,
                        "fp1_laps": fp1_laps,
                        "fp2_laps": fp2_laps,
                        "fp3_laps": fp3_laps,
                    }
                )

                driver_points[abb] = prev_points + points_awarded
                driver_recent_points[abb].append(points_awarded)
                if team:
                    team_points[team] = team_prev_points + points_awarded

            loaded_events += 1
            iterator.set_postfix({"loaded": f"{loaded_events}/{total_events}"})

        print(f"{year}: loaded {loaded_events}/{total_events} events")

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
                "year",
                "round_number",
                "grid_position",
                "quali_delta",
                "quali_tm_delta",
                "season_pts_driver",
                "season_pts_team",
                "last_3_avg",
                "is_street_circuit",
                "is_wet",
                "race_time",
                "race_status",
                "gap_to_winner",
                "laps",
                "race_time_per_lap",
                "fp1_best",
                "fp2_best",
                "fp3_best",
                "fp1_rel",
                "fp2_rel",
                "fp3_rel",
                "fp1_laps",
                "fp2_laps",
                "fp3_laps",
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    quiet_logs()
    enable_cache(args.cache_dir)

    df = build_dataset(args.years, args.session_type, args.limit_rounds)
    if df.empty:
        print("No data collected; check the chosen years/rounds.")
        return

    feature_cols = [
        "driver_id",
        "constructor_id",
        "circuit_id",
        "year",
        "round_number",
        "grid_position",
        "quali_delta",
        "quali_tm_delta",
        "season_pts_driver",
        "season_pts_team",
        "last_3_avg",
        "is_street_circuit",
        "is_wet",
        "laps",
        "fp1_best",
        "fp2_best",
        "fp3_best",
        "fp1_rel",
        "fp2_rel",
        "fp3_rel",
        "fp1_laps",
        "fp2_laps",
        "fp3_laps",
    ]
    target_col = "race_time"
    print(
        "Built dataset with "
        f"{len(df)} rows. Feature columns: {feature_cols}. Target: {target_col}."
    )

    df["year"] = df["year"].fillna(0).astype(int)
    df["round_number"] = df["round_number"].fillna(0).astype(int)
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
    df["race_status"] = df["race_status"].fillna("")
    df["laps"] = df["laps"].fillna(0).astype(int)
    for col in [
        "fp1_best",
        "fp2_best",
        "fp3_best",
        "fp1_rel",
        "fp2_rel",
        "fp3_rel",
        "fp1_laps",
        "fp2_laps",
        "fp3_laps",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col.endswith("_laps"):
                df[col] = df[col].fillna(0).astype(int)
            else:
                non_null = df[col].dropna()
                fill_value = float(non_null.median()) if not non_null.empty else 0.0
                df[col] = df[col].fillna(fill_value)
    if "race_time" in df.columns:
        df["race_time"] = pd.to_numeric(df["race_time"], errors="coerce")
        finite = df["race_time"].dropna()
        if not finite.empty:
            lower, upper = finite.quantile([0.01, 0.99])
            df["race_time"] = df["race_time"].clip(lower=lower, upper=upper)
    if "gap_to_winner" in df.columns:
        df["gap_to_winner"] = pd.to_numeric(df["gap_to_winner"], errors="coerce")
        finite_gap = df["gap_to_winner"].dropna()
        if not finite_gap.empty:
            gl, gu = finite_gap.quantile([0.01, 0.99])
            df["gap_to_winner"] = df["gap_to_winner"].clip(lower=gl, upper=gu)
    if "race_time_per_lap" in df.columns:
        df["race_time_per_lap"] = pd.to_numeric(df["race_time_per_lap"], errors="coerce")
        finite_lap = df["race_time_per_lap"].dropna()
        if not finite_lap.empty:
            ll, lu = finite_lap.quantile([0.01, 0.99])
            df["race_time_per_lap"] = df["race_time_per_lap"].clip(lower=ll, upper=lu)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
