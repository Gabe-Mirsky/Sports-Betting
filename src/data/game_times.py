"""Free NBA game start-time helpers."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .scoreboard import fetch_scoreboard_games
from .team_aliases import normalize_team_abbr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESPN_NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
EASTERN = ZoneInfo("America/New_York")


def _parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def fetch_espn_scoreboard_date(game_date: str | pd.Timestamp, timeout: int = 30) -> pd.DataFrame:
    """Fetch one date of free ESPN NBA scoreboard data and return game start times."""

    date_value = pd.Timestamp(game_date)
    params = urllib.parse.urlencode({"dates": date_value.strftime("%Y%m%d")})
    url = f"{ESPN_NBA_SCOREBOARD_URL}?{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competitors = competitions[0].get("competitors") or []
        home_team = ""
        away_team = ""
        for competitor in competitors:
            team = competitor.get("team") or {}
            abbr = normalize_team_abbr(team.get("abbreviation", ""))
            if competitor.get("homeAway") == "home":
                home_team = abbr
            elif competitor.get("homeAway") == "away":
                away_team = abbr
        if not home_team or not away_team:
            continue
        rows.append(
            {
                "game_date": date_value.date().isoformat(),
                "home_team_abbr": home_team,
                "away_team_abbr": away_team,
                "game_start_time": event.get("date", ""),
                "game_time_source": "espn_scoreboard",
                "espn_event_id": event.get("id", ""),
            }
        )
    return pd.DataFrame(rows)


def _official_start_time_to_utc(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(EASTERN)
    return timestamp.tz_convert("UTC").isoformat()


def fetch_nba_official_scoreboard_date(game_date: str | pd.Timestamp, timeout: int = 30) -> pd.DataFrame:
    """Fetch one date of NBA Stats scoreboard data and return official game start times."""

    date_value = pd.Timestamp(game_date)
    games = fetch_scoreboard_games(
        date_value.date().isoformat(),
        timeout=timeout,
        retries=1,
        log_warnings=False,
    )
    if games.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in games.iterrows():
        home_team = normalize_team_abbr(row.get("home_team_abbr", ""))
        away_team = normalize_team_abbr(row.get("away_team_abbr", ""))
        start_time = _official_start_time_to_utc(row.get("game_date"))
        if not home_team or not away_team or not start_time:
            continue
        rows.append(
            {
                "game_date": date_value.date().isoformat(),
                "home_team_abbr": home_team,
                "away_team_abbr": away_team,
                "game_start_time": start_time,
                "game_time_source": "nba_stats_scoreboard",
                "nba_game_id": row.get("game_id", ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_game_start_times_for_date(game_date: str | pd.Timestamp, timeout: int = 30) -> pd.DataFrame:
    """Prefer official NBA Stats start times, with ESPN as a free fallback."""

    try:
        official = fetch_nba_official_scoreboard_date(game_date, timeout=timeout)
        if not official.empty:
            return official
    except Exception:
        pass
    fallback = fetch_espn_scoreboard_date(game_date, timeout=timeout)
    if not fallback.empty:
        fallback = fallback.copy()
        fallback["game_time_source"] = "espn_scoreboard_after_nba_stats_fallback"
    return fallback


def download_game_start_times_for_games(
    games_df: pd.DataFrame,
    output_path: str | Path | None = None,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Download start times for the unique dates in a game dataframe."""

    games = games_df.copy()
    games["game_date"] = _parse_dates(games["game_date"]).dt.normalize()
    dates = sorted(games["game_date"].dropna().dt.date.unique())
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for game_date in dates:
        try:
            frame = fetch_game_start_times_for_date(str(game_date))
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            failures.append({"game_date": str(game_date), "error": f"{type(exc).__name__}: {exc}"})
        time.sleep(sleep_seconds)

    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not output.empty:
        output = output.drop_duplicates(subset=["game_date", "home_team_abbr", "away_team_abbr"], keep="last")
    path = Path(output_path) if output_path else PROJECT_ROOT / "data" / "interim" / "nba_game_start_times.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not output.empty or not path.exists():
        output.to_csv(path, index=False)
    if failures:
        failure_path = (
            path.with_name(f"{path.stem}_failures.csv")
            if output_path
            else PROJECT_ROOT / "data" / "reports" / "nba_game_start_time_failures.csv"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    return output


def add_game_start_times(games_df: pd.DataFrame, starts_df: pd.DataFrame) -> pd.DataFrame:
    """Merge downloaded start times into game rows by date and teams."""

    games = games_df.copy()
    starts = starts_df.copy()
    games = games.drop(
        columns=[
            column
            for column in ["game_start_time", "game_time_source", "espn_event_id", "nba_game_id"]
            if column in games.columns
        ]
    )
    games["game_date"] = _parse_dates(games["game_date"]).dt.normalize()
    starts["game_date"] = _parse_dates(starts["game_date"]).dt.normalize()
    for column in ["home_team_abbr", "away_team_abbr"]:
        games[column] = games[column].map(normalize_team_abbr)
        starts[column] = starts[column].map(normalize_team_abbr)
    return games.merge(
        starts,
        on=["game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
    )
