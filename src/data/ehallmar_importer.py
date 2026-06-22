"""Importer for the ehallmar NBA historical stats + betting Kaggle dataset.

Normalizes the dataset into three clean tables (games, player game logs, game-level
odds) plus a summary. This dataset has NBA game results, multi-book *game* odds, and
player game-stat actuals joinable by ``game_id``. It does NOT contain player-prop
betting lines, so it is labeled ``player_actuals_source`` (not a prop market source).

Inspection / ingestion only: no model logic, no proof-gate or betting changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE = "ehallmar"
SOURCE_ROLE = "player_actuals_source"  # NOT a player-prop market source.
SPORT = "basketball"
LEAGUE = "NBA"

GAMES_FILE = "nba_games_all.csv"
TEAMS_FILE = "nba_teams_all.csv"
PLAYER_STATS_FILE = "nba_players_game_stats.csv"
ODDS_FILES = {
    "moneyline": "nba_betting_money_line.csv",
    "spread": "nba_betting_spread.csv",
    "total": "nba_betting_totals.csv",
}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs) if path.exists() else pd.DataFrame()


def _team_id_to_abbr(dataset_dir: Path) -> dict[int, str]:
    teams = _read_csv(dataset_dir / TEAMS_FILE)
    if teams.empty or not {"team_id", "abbreviation"}.issubset(teams.columns):
        return {}
    return {
        int(row["team_id"]): str(row["abbreviation"]).strip()
        for _, row in teams.iterrows()
        if pd.notna(row["team_id"])
    }


def _game_key(date: pd.Series, home: pd.Series, away: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date, errors="coerce").dt.date.astype(str)
    return dates + "|" + home.astype(str) + "|" + away.astype(str)


def normalize_games(dataset_dir: Path, min_season: int | None = None) -> pd.DataFrame:
    """Pivot the two-rows-per-game table into one row per game with home/away scores."""

    raw = _read_csv(dataset_dir / GAMES_FILE)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw["is_home_bool"] = raw["is_home"].astype(str).str.lower().str.startswith("t")
    abbr = _team_id_to_abbr(dataset_dir)

    home = raw[raw["is_home_bool"]].drop_duplicates("game_id", keep="first")
    away = raw[~raw["is_home_bool"]].drop_duplicates("game_id", keep="first")
    home = home[["game_id", "game_date", "team_id", "pts", "wl", "season", "season_year", "season_type"]].rename(
        columns={"team_id": "home_team_id", "pts": "home_score", "wl": "home_wl"}
    )
    away = away[["game_id", "team_id", "pts"]].rename(columns={"team_id": "away_team_id", "pts": "away_score"})
    games = home.merge(away, on="game_id", how="inner")

    games["home_team_abbr"] = games["home_team_id"].map(abbr).fillna("")
    games["away_team_abbr"] = games["away_team_id"].map(abbr).fillna("")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    games["home_win"] = games["home_wl"].astype(str).str.upper().eq("W")
    games["winner"] = np.where(games["home_win"], "home", "away")
    games["season_start_year"] = pd.to_numeric(games["season_year"], errors="coerce").astype("Int64")
    games["game_key"] = _game_key(games["game_date"], games["home_team_abbr"], games["away_team_abbr"])
    games["source"] = SOURCE
    games["source_role"] = SOURCE_ROLE
    games["sport"] = SPORT
    games["league"] = LEAGUE

    if min_season is not None:
        games = games[games["season_start_year"].fillna(-1).astype(int) >= int(min_season)]
    columns = [
        "source", "source_role", "sport", "league", "game_id", "game_date", "season", "season_type",
        "season_start_year", "home_team_id", "away_team_id", "home_team_abbr", "away_team_abbr",
        "home_score", "away_score", "home_win", "winner", "game_key",
    ]
    return games[columns].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def normalize_player_game_logs(dataset_dir: Path, min_season: int | None = None) -> pd.DataFrame:
    """Normalize player box-score actuals to the project's player_game_log columns."""

    raw = _read_csv(dataset_dir / PLAYER_STATS_FILE)
    if raw.empty:
        return pd.DataFrame()
    matchup = raw["matchup"].astype(str).str.upper()
    opponent = matchup.str.extract(r"(?:VS\.?|@)\s*([A-Z0-9]{2,4})\s*$")[0]
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "source_role": SOURCE_ROLE,
            "sport": SPORT,
            "league": LEAGUE,
            "player_id": raw.get("player_id"),
            "player_name": raw.get("player_name"),
            "game_id": raw.get("game_id"),
            "game_date": pd.to_datetime(raw.get("game_date"), errors="coerce").dt.normalize(),
            "season": raw.get("season"),
            "season_start_year": pd.to_numeric(raw.get("season_year"), errors="coerce").astype("Int64"),
            "team_abbr": raw.get("team_abbreviation"),
            "opponent_abbr": opponent,
            "is_home": matchup.str.contains("VS", regex=False),
            "minutes": pd.to_numeric(raw.get("min"), errors="coerce"),
            "points": pd.to_numeric(raw.get("pts"), errors="coerce"),
            "rebounds": pd.to_numeric(raw.get("reb"), errors="coerce"),
            "assists": pd.to_numeric(raw.get("ast"), errors="coerce"),
            "threes": pd.to_numeric(raw.get("fg3m"), errors="coerce"),
            "blocks": pd.to_numeric(raw.get("blk"), errors="coerce"),
            "steals": pd.to_numeric(raw.get("stl"), errors="coerce"),
            "turnovers": pd.to_numeric(raw.get("tov"), errors="coerce"),
        }
    )
    if min_season is not None:
        out = out[out["season_start_year"].fillna(-1).astype(int) >= int(min_season)]
    return out.reset_index(drop=True)


def normalize_game_odds(dataset_dir: Path) -> pd.DataFrame:
    """Stack moneyline/spread/total book odds into one long game-market table."""

    abbr = _team_id_to_abbr(dataset_dir)
    frames: list[pd.DataFrame] = []
    for market_type, filename in ODDS_FILES.items():
        raw = _read_csv(dataset_dir / filename)
        if raw.empty:
            continue
        block = pd.DataFrame(
            {
                "source": SOURCE,
                "source_role": SOURCE_ROLE,
                "sport": SPORT,
                "league": LEAGUE,
                "market_type": market_type,
                "market_scope": "game_market",
                "game_id": raw.get("game_id"),
                "book_name": raw.get("book_name"),
                "book_id": raw.get("book_id"),
                "team_abbr": raw.get("team_id").map(abbr) if "team_id" in raw else "",
                "opponent_abbr": raw.get("a_team_id").map(abbr) if "a_team_id" in raw else "",
                "side1_price": pd.to_numeric(raw.get("price1"), errors="coerce"),
                "side2_price": pd.to_numeric(raw.get("price2"), errors="coerce"),
                "price_format": "american",
            }
        )
        block["side1_line"] = pd.to_numeric(raw.get("spread1", raw.get("total1")), errors="coerce") if market_type != "moneyline" else np.nan
        block["side2_line"] = pd.to_numeric(raw.get("spread2", raw.get("total2")), errors="coerce") if market_type != "moneyline" else np.nan
        frames.append(block)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def import_ehallmar(
    dataset_dir: str | Path,
    processed_dir: str | Path,
    reports_dir: str | Path,
    min_season: int | None = None,
) -> dict[str, Any]:
    """Run the full ehallmar import and write normalized CSVs + a summary JSON."""

    dataset_dir = Path(dataset_dir)
    processed = Path(processed_dir)
    reports = Path(reports_dir)
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    games = normalize_games(dataset_dir, min_season=min_season)
    player_logs = normalize_player_game_logs(dataset_dir, min_season=min_season)
    odds = normalize_game_odds(dataset_dir)

    games_path = processed / "nba_games_normalized.csv"
    logs_path = processed / "nba_player_game_logs_normalized.csv"
    odds_path = processed / "nba_game_odds_normalized.csv"
    games.to_csv(games_path, index=False)
    player_logs.to_csv(logs_path, index=False)
    odds.to_csv(odds_path, index=False)

    summary = _build_summary(dataset_dir, games, player_logs, odds)
    (reports / "ehallmar_import_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _date_range(frame: pd.DataFrame, column: str = "game_date") -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {"min": None, "max": None}
    dates = pd.to_datetime(frame[column], errors="coerce")
    return {"min": _iso(dates.min()), "max": _iso(dates.max())}


def _iso(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _build_summary(dataset_dir: Path, games: pd.DataFrame, logs: pd.DataFrame, odds: pd.DataFrame) -> dict[str, Any]:
    odds_by_market = (
        {str(k): int(v) for k, v in odds["market_type"].value_counts().items()} if not odds.empty else {}
    )
    return {
        "source": SOURCE,
        "source_role": SOURCE_ROLE,
        "has_player_prop_lines": False,
        "market_scope": "game_market",
        "dataset_dir": str(dataset_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "games": {
            "rows": int(len(games)),
            "unique_game_ids": int(games["game_id"].nunique()) if not games.empty else 0,
            "with_scores": int(games[["home_score", "away_score"]].notna().all(axis=1).sum()) if not games.empty else 0,
            "date_range": _date_range(games),
            "distinct_teams": int(pd.unique(games[["home_team_abbr", "away_team_abbr"]].values.ravel()).size) if not games.empty else 0,
        },
        "player_game_logs": {
            "rows": int(len(logs)),
            "unique_players": int(logs["player_id"].nunique()) if not logs.empty else 0,
            "unique_game_ids": int(logs["game_id"].nunique()) if not logs.empty else 0,
            "missing_player_name": int(logs["player_name"].isna().sum()) if not logs.empty else 0,
            "date_range": _date_range(logs),
        },
        "game_odds": {
            "rows": int(len(odds)),
            "by_market_type": odds_by_market,
            "unique_books": int(odds["book_name"].nunique()) if not odds.empty else 0,
            "unique_game_ids": int(odds["game_id"].nunique()) if not odds.empty else 0,
        },
        "join_keys": ["game_id", "game_key (game_date|home_team_abbr|away_team_abbr)"],
        "notes": [
            "Player game logs SETTLE props (actuals); they are not prop market lines.",
            "Game odds are team game markets (moneyline/spread/total), not player props.",
            "Odds prices are American; player-stat history extends decades before the odds coverage.",
        ],
        "research_only": True,
        "approved": False,
    }
