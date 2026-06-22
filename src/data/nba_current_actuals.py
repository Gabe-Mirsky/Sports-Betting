"""Current/recent NBA actuals importer built on the cached nba_api downloads.

Normalizes the raw nba_api LeagueGameLog caches (team + player) for recent
seasons into the project's canonical shapes:

- one row per game with home/away teams, final scores, and canonical game key
- one row per player-game with minutes/points/rebounds/assists/threes/steals/
  blocks/turnovers actuals and canonical game key

These are ACTUALS for settling player props - not market lines. Ingestion only:
no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .canonical_games import GAME_KEY_VERSION, canonical_game_key_series
from .team_aliases import normalize_team_abbr


SOURCE = "nba_api"
SOURCE_ROLE = "player_actuals_source"
SPORT = "basketball"
LEAGUE = "NBA"

DEFAULT_MIN_SEASON = 2024

_PLAYER_STAT_MAP = {
    "MIN": "minutes",
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "FG3M": "threes",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
}


def _season_from_filename(name: str) -> tuple[int | None, str]:
    """Extract (season_start_year, season_type) from a cache filename."""

    match = re.search(r"game_log_(\d{4})(_playoffs)?", name)
    if not match:
        return None, "Regular Season"
    season_type = "Playoffs" if match.group(2) else "Regular Season"
    return int(match.group(1)), season_type


def discover_cached_files(raw_dir: str | Path, kind: str) -> list[Path]:
    """Find cached nba_api game-log files of a kind ('team' or 'player').

    Team caches live under both naming schemes used by this repo:
    ``data/raw/nba/league_game_log_*.parquet`` and
    ``data/raw/nba_league_game_log_*.parquet``; player caches under
    ``data/raw/nba/player/player_game_log_*``.
    """

    raw = Path(raw_dir)
    if kind == "player":
        patterns = [raw / "nba" / "player"]
        prefix = "player_game_log_"
    else:
        patterns = [raw / "nba", raw]
        prefix = "league_game_log_"
    files: dict[str, Path] = {}
    for folder in patterns:
        if not folder.exists():
            continue
        for path in sorted(folder.glob(f"*{prefix}*")):
            if path.suffix.lower() in {".parquet", ".csv"}:
                files.setdefault(path.stem.replace("nba_", ""), path)
    return sorted(files.values(), key=lambda p: p.name)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def load_raw_logs(raw_dir: str | Path, kind: str, min_season: int | None = None) -> pd.DataFrame:
    """Load and concatenate cached nba_api logs of a kind for recent seasons."""

    frames: list[pd.DataFrame] = []
    for path in discover_cached_files(raw_dir, kind):
        season_year, season_type = _season_from_filename(path.name)
        if season_year is None:
            continue
        if min_season is not None and season_year < min_season:
            continue
        frame = _read_table(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["season_start_year"] = season_year
        if "season_type" not in frame.columns:
            frame["season_type"] = season_type
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_current_games(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Pivot nba_api team game logs (2 rows/game) into one row per game."""

    if team_logs.empty:
        return pd.DataFrame()
    logs = team_logs.copy()
    matchup = logs["MATCHUP"].astype(str).str.upper()
    logs["is_home"] = matchup.str.contains("VS", regex=False)
    logs["team_abbr"] = logs["TEAM_ABBREVIATION"].map(normalize_team_abbr)

    home = logs[logs["is_home"]].drop_duplicates("GAME_ID", keep="first")
    away = logs[~logs["is_home"]].drop_duplicates("GAME_ID", keep="first")
    home = home[
        ["GAME_ID", "GAME_DATE", "team_abbr", "TEAM_ID", "TEAM_NAME", "PTS", "WL",
         "nba_season", "season_type", "season_start_year"]
    ].rename(
        columns={
            "team_abbr": "home_team_abbr",
            "TEAM_ID": "home_team_id",
            "TEAM_NAME": "home_team_name",
            "PTS": "home_score",
            "WL": "home_wl",
        }
    )
    away = away[["GAME_ID", "team_abbr", "TEAM_ID", "TEAM_NAME", "PTS"]].rename(
        columns={
            "team_abbr": "away_team_abbr",
            "TEAM_ID": "away_team_id",
            "TEAM_NAME": "away_team_name",
            "PTS": "away_score",
        }
    )
    games = home.merge(away, on="GAME_ID", how="inner")
    games["game_date"] = pd.to_datetime(games["GAME_DATE"], errors="coerce").dt.normalize()
    games["home_win"] = games["home_wl"].astype(str).str.upper().eq("W")
    games["winner"] = games["home_win"].map({True: "home", False: "away"})
    games = games.rename(columns={"GAME_ID": "game_id", "nba_season": "season"})
    games["source"] = SOURCE
    games["source_role"] = SOURCE_ROLE
    games["sport"] = SPORT
    games["league"] = LEAGUE
    games["canonical_game_key"] = canonical_game_key_series(
        games, SPORT, "league", "game_date", "home_team_abbr", "away_team_abbr"
    )
    games["key_version"] = GAME_KEY_VERSION

    columns = [
        "source", "source_role", "sport", "league", "game_id", "game_date", "season",
        "season_type", "season_start_year", "home_team_id", "away_team_id",
        "home_team_abbr", "away_team_abbr", "home_team_name", "away_team_name",
        "home_score", "away_score", "home_win", "winner", "canonical_game_key", "key_version",
    ]
    return games[columns].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def normalize_current_player_logs(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize nba_api player game logs to the project player_game_log shape."""

    if player_logs.empty:
        return pd.DataFrame()
    logs = player_logs.copy()
    matchup = logs["MATCHUP"].astype(str).str.upper()
    opponent_raw = matchup.str.extract(r"(?:VS\.?|@)\s*([A-Z0-9]{2,4})\s*$")[0]
    is_home = matchup.str.contains("VS", regex=False)
    team_abbr = logs["TEAM_ABBREVIATION"].map(normalize_team_abbr)
    opponent_abbr = opponent_raw.map(normalize_team_abbr)

    out = pd.DataFrame(
        {
            "source": SOURCE,
            "source_role": SOURCE_ROLE,
            "sport": SPORT,
            "league": LEAGUE,
            "player_id": logs.get("PLAYER_ID"),
            "player_name": logs.get("PLAYER_NAME"),
            "team_id": logs.get("TEAM_ID"),
            "team_abbr": team_abbr,
            "team_name": logs.get("TEAM_NAME"),
            "opponent_abbr": opponent_abbr,
            "is_home": is_home,
            "game_id": logs.get("GAME_ID"),
            "game_date": pd.to_datetime(logs.get("GAME_DATE"), errors="coerce").dt.normalize(),
            "season": logs.get("nba_season"),
            "season_type": logs.get("season_type"),
            "season_start_year": pd.to_numeric(logs.get("season_start_year"), errors="coerce").astype("Int64"),
        }
    )
    for raw_col, out_col in _PLAYER_STAT_MAP.items():
        out[out_col] = pd.to_numeric(logs.get(raw_col), errors="coerce")

    out["home_team_abbr"] = out["team_abbr"].where(out["is_home"], out["opponent_abbr"])
    out["away_team_abbr"] = out["opponent_abbr"].where(out["is_home"], out["team_abbr"])
    out["canonical_game_key"] = canonical_game_key_series(
        out, SPORT, "league", "game_date", "home_team_abbr", "away_team_abbr"
    )
    out["key_version"] = GAME_KEY_VERSION
    out = out.drop(columns=["home_team_abbr", "away_team_abbr"])
    return out.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def import_nba_current_actuals(
    raw_dir: str | Path,
    processed_dir: str | Path,
    reports_dir: str | Path,
    min_season: int = DEFAULT_MIN_SEASON,
) -> dict[str, Any]:
    """Run the current NBA actuals import; write normalized CSVs + summary JSON."""

    processed = Path(processed_dir)
    reports = Path(reports_dir)
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    team_logs = load_raw_logs(raw_dir, "team", min_season=min_season)
    player_logs = load_raw_logs(raw_dir, "player", min_season=min_season)

    games = normalize_current_games(team_logs)
    players = normalize_current_player_logs(player_logs)

    games.to_csv(processed / "nba_current_games_normalized.csv", index=False)
    players.to_csv(processed / "nba_current_player_game_logs_normalized.csv", index=False)

    summary = _build_summary(games, players, min_season)
    (reports / "nba_current_actuals_import_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _date_range(frame: pd.DataFrame, column: str = "game_date") -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {"min": None, "max": None}
    dates = pd.to_datetime(frame[column], errors="coerce")
    if dates.isna().all():
        return {"min": None, "max": None}
    return {"min": dates.min().date().isoformat(), "max": dates.max().date().isoformat()}


def _build_summary(games: pd.DataFrame, players: pd.DataFrame, min_season: int) -> dict[str, Any]:
    stat_coverage = {}
    if not players.empty:
        for column in _PLAYER_STAT_MAP.values():
            stat_coverage[column] = int(players[column].notna().sum())
    return {
        "report": "nba_current_actuals_import_summary",
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "sport": SPORT,
        "league": LEAGUE,
        "min_season_start_year": int(min_season),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "games": {
            "rows": int(len(games)),
            "unique_game_ids": int(games["game_id"].nunique()) if not games.empty else 0,
            "with_final_scores": int(games[["home_score", "away_score"]].notna().all(axis=1).sum()) if not games.empty else 0,
            "seasons": sorted({str(s) for s in games["season"].dropna()}) if not games.empty else [],
            "by_season_type": {str(k): int(v) for k, v in games["season_type"].value_counts().items()} if not games.empty else {},
            "date_range": _date_range(games),
            "rows_with_invalid_key": int(games["canonical_game_key"].astype(str).eq("").sum()) if not games.empty else 0,
            "duplicate_canonical_keys": int(games["canonical_game_key"].duplicated().sum()) if not games.empty else 0,
        },
        "player_game_logs": {
            "rows": int(len(players)),
            "unique_players": int(players["player_id"].nunique()) if not players.empty else 0,
            "unique_game_ids": int(players["game_id"].nunique()) if not players.empty else 0,
            "date_range": _date_range(players),
            "stat_coverage_non_null": stat_coverage,
            "rows_with_invalid_key": int(players["canonical_game_key"].astype(str).eq("").sum()) if not players.empty else 0,
        },
        "notes": [
            "Built from the cached nba_api LeagueGameLog downloads (team + player).",
            "Player rows are ACTUALS for settling props - not market lines.",
            "canonical_game_key joins these to zachht/Kalshi/sportsbook odds by date+teams.",
        ],
        "research_only": True,
        "approved": False,
    }
