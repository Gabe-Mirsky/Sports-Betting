"""Historical sportsbook odds loading and matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.seasons import nba_season_display_label, season_start_year_from_dates
from data.team_aliases import normalize_team_abbr


DATE_ALIASES = ["game_date", "date", "commence_time", "timestamp"]
HOME_TEAM_ALIASES = ["home_team_abbr", "home_team", "home", "home_team_name"]
AWAY_TEAM_ALIASES = ["away_team_abbr", "away_team", "away", "away_team_name"]
HOME_MONEYLINE_ALIASES = ["home_moneyline", "home_ml", "home_odds", "home_close", "close_home_moneyline"]
AWAY_MONEYLINE_ALIASES = ["away_moneyline", "away_ml", "away_odds", "away_close", "close_away_moneyline"]
SPORTSBOOK_ALIASES = ["sportsbook_name", "sportsbook", "book", "bookmaker"]
TIMESTAMP_ALIASES = ["closing_timestamp", "timestamp", "last_update", "updated_at"]
SPREAD_ALIASES = ["spread", "home_spread", "closing_spread"]
TOTAL_ALIASES = ["total", "closing_total"]
HOME_SCORE_ALIASES = ["home_score", "home_final", "home_points"]
AWAY_SCORE_ALIASES = ["away_score", "away_final", "away_points"]


def moneyline_to_implied_probability(value: Any) -> float:
    """Convert American moneyline odds to implied probability."""

    odds = float(value)
    if np.isnan(odds) or odds == 0:
        return float("nan")
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def _first_present(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _canonical_game_key(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["game_date"], errors="coerce").dt.date.astype(str)
    return dates + "|" + frame["home_team_abbr"].astype(str) + "|" + frame["away_team_abbr"].astype(str)


def add_sportsbook_game_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized NBA team abbreviations, season, and game_key."""

    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce").dt.normalize()
    if "home_team_abbr" in output.columns:
        output["home_team_abbr"] = output["home_team_abbr"].map(normalize_team_abbr)
    elif "home_team" in output.columns:
        output["home_team_abbr"] = output["home_team"].map(normalize_team_abbr)
    if "away_team_abbr" in output.columns:
        output["away_team_abbr"] = output["away_team_abbr"].map(normalize_team_abbr)
    elif "away_team" in output.columns:
        output["away_team_abbr"] = output["away_team"].map(normalize_team_abbr)
    if "season" not in output.columns:
        output["season"] = season_start_year_from_dates(output["game_date"]).astype("Int64")
    else:
        output["season"] = pd.to_numeric(output["season"], errors="coerce").fillna(
            season_start_year_from_dates(output["game_date"])
        ).astype("Int64")
    output["game_key"] = _canonical_game_key(output)
    return output


def normalize_sportsbook_odds(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a sportsbook odds file to canonical columns with implied probabilities."""

    columns = list(raw.columns)
    date_column = _first_present(columns, DATE_ALIASES)
    home_column = _first_present(columns, HOME_TEAM_ALIASES)
    away_column = _first_present(columns, AWAY_TEAM_ALIASES)
    home_ml_column = _first_present(columns, HOME_MONEYLINE_ALIASES)
    away_ml_column = _first_present(columns, AWAY_MONEYLINE_ALIASES)
    missing = [
        name
        for name, column in [
            ("game_date", date_column),
            ("home_team", home_column),
            ("away_team", away_column),
            ("home_moneyline", home_ml_column),
            ("away_moneyline", away_ml_column),
        ]
        if column is None
    ]
    if missing:
        raise ValueError(f"Sportsbook odds file is missing required columns: {missing}")

    output = pd.DataFrame(
        {
            "game_date": pd.to_datetime(raw[date_column], errors="coerce").dt.normalize(),
            "home_team_abbr": raw[home_column].map(normalize_team_abbr),
            "away_team_abbr": raw[away_column].map(normalize_team_abbr),
            "home_moneyline": pd.to_numeric(raw[home_ml_column], errors="coerce"),
            "away_moneyline": pd.to_numeric(raw[away_ml_column], errors="coerce"),
        }
    )
    sportsbook_column = _first_present(columns, SPORTSBOOK_ALIASES)
    timestamp_column = _first_present(columns, TIMESTAMP_ALIASES)
    spread_column = _first_present(columns, SPREAD_ALIASES)
    total_column = _first_present(columns, TOTAL_ALIASES)
    home_score_column = _first_present(columns, HOME_SCORE_ALIASES)
    away_score_column = _first_present(columns, AWAY_SCORE_ALIASES)
    output["sportsbook_name"] = raw[sportsbook_column].astype(str) if sportsbook_column else ""
    output["closing_timestamp"] = pd.to_datetime(raw[timestamp_column], errors="coerce") if timestamp_column else pd.NaT
    output["spread"] = pd.to_numeric(raw[spread_column], errors="coerce") if spread_column else np.nan
    output["total"] = pd.to_numeric(raw[total_column], errors="coerce") if total_column else np.nan
    output["home_score"] = pd.to_numeric(raw[home_score_column], errors="coerce") if home_score_column else np.nan
    output["away_score"] = pd.to_numeric(raw[away_score_column], errors="coerce") if away_score_column else np.nan
    if "is_closing" in raw.columns:
        output["is_closing"] = raw["is_closing"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    elif "closing_odds_flag" in raw.columns:
        output["is_closing"] = raw["closing_odds_flag"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    else:
        output["is_closing"] = output["closing_timestamp"].notna()

    output = output.dropna(subset=["game_date", "home_moneyline", "away_moneyline"]).copy()
    output["home_implied_prob"] = output["home_moneyline"].map(moneyline_to_implied_probability)
    output["away_implied_prob"] = output["away_moneyline"].map(moneyline_to_implied_probability)
    overround = output["home_implied_prob"] + output["away_implied_prob"]
    output["home_no_vig_prob"] = output["home_implied_prob"] / overround
    output["away_no_vig_prob"] = output["away_implied_prob"] / overround
    output.loc[~np.isfinite(overround) | (overround <= 0), ["home_no_vig_prob", "away_no_vig_prob"]] = np.nan
    output = add_sportsbook_game_keys(output)
    return output.sort_values(["game_date", "home_team_abbr", "away_team_abbr"]).reset_index(drop=True)


def load_sportsbook_odds(path: str | Path) -> pd.DataFrame:
    """Load historical sportsbook odds from CSV/Parquet/JSON."""

    odds_path = Path(path)
    if not odds_path.exists():
        return pd.DataFrame()
    if odds_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(odds_path)
    elif odds_path.suffix.lower() in {".json", ".jsonl"}:
        raw = pd.read_json(odds_path, lines=odds_path.suffix.lower() == ".jsonl")
    else:
        raw = pd.read_csv(odds_path, low_memory=False)
    return normalize_sportsbook_odds(raw)


def select_closing_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Select one closing/preferred odds row per game."""

    if odds.empty:
        return odds.copy()
    working = odds.copy()
    working["closing_rank"] = (~working["is_closing"].fillna(False)).astype(int)
    working["timestamp_rank"] = pd.to_datetime(working["closing_timestamp"], errors="coerce").fillna(pd.Timestamp.min)
    return (
        working.sort_values(["game_key", "closing_rank", "timestamp_rank"], ascending=[True, True, False])
        .drop_duplicates("game_key", keep="first")
        .drop(columns=["closing_rank", "timestamp_rank"])
        .reset_index(drop=True)
    )


def match_sportsbook_odds_to_games(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Attach selected sportsbook odds to game rows by date and teams."""

    if games.empty or odds.empty:
        return pd.DataFrame()
    game_frame = add_sportsbook_game_keys(games)
    selected = select_closing_odds(odds)
    matched = game_frame.merge(
        selected.drop(columns=["game_date", "home_team_abbr", "away_team_abbr", "season"], errors="ignore"),
        on="game_key",
        how="left",
    )
    return matched.reset_index(drop=True)


def sportsbook_match_report_by_season(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Return per-season NBA/sportsbook matching diagnostics."""

    columns = [
        "season",
        "season_label",
        "nba_games",
        "sportsbook_rows",
        "matched_games",
        "unmatched_nba_games",
        "unmatched_sportsbook_games",
        "match_rate",
    ]
    if games.empty:
        return pd.DataFrame(columns=columns)

    game_frame = add_sportsbook_game_keys(games).dropna(subset=["game_date", "home_team_abbr", "away_team_abbr"])
    game_keys = game_frame[["season", "game_key"]].drop_duplicates()
    selected_odds = select_closing_odds(odds) if not odds.empty else odds.copy()
    odds_keys = (
        add_sportsbook_game_keys(selected_odds)[["season", "game_key"]].drop_duplicates()
        if not selected_odds.empty
        else pd.DataFrame(columns=["season", "game_key"])
    )
    odds_rows = (
        add_sportsbook_game_keys(odds)[["season", "game_key"]]
        if not odds.empty
        else pd.DataFrame(columns=["season", "game_key"])
    )

    seasons = sorted(
        {
            int(season)
            for season in pd.concat([game_keys["season"], odds_keys["season"]], ignore_index=True).dropna().unique()
        }
    )
    rows: list[dict[str, Any]] = []
    for season in seasons:
        nba_season = game_keys[game_keys["season"].eq(season)]
        odds_season = odds_keys[odds_keys["season"].eq(season)]
        row_odds_season = odds_rows[odds_rows["season"].eq(season)]
        nba_set = set(nba_season["game_key"].dropna())
        odds_set = set(odds_season["game_key"].dropna())
        matched = nba_set & odds_set
        nba_games = len(nba_set)
        rows.append(
            {
                "season": season,
                "season_label": nba_season_display_label(season),
                "nba_games": nba_games,
                "sportsbook_rows": int(len(row_odds_season)),
                "matched_games": int(len(matched)),
                "unmatched_nba_games": int(len(nba_set - odds_set)),
                "unmatched_sportsbook_games": int(len(odds_set - nba_set)),
                "match_rate": len(matched) / nba_games if nba_games else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("season").reset_index(drop=True)


def sportsbook_coverage_by_season(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Return NBA game and sportsbook odds coverage by season."""

    if games.empty:
        return pd.DataFrame(columns=["season", "season_label", "nba_games", "sportsbook_games", "sportsbook_match_rate"])
    matched = match_sportsbook_odds_to_games(games, odds) if not odds.empty else games.copy()
    matched["season"] = pd.to_numeric(matched["season"], errors="coerce")
    if "game_id" in matched.columns:
        game_counts = matched.groupby("season")["game_id"].nunique()
        with_odds = matched[matched.get("home_no_vig_prob", pd.Series(np.nan, index=matched.index)).notna()]
        odds_counts = with_odds.groupby("season")["game_id"].nunique()
    else:
        matched["game_key"] = _canonical_game_key(matched)
        game_counts = matched.groupby("season")["game_key"].nunique()
        with_odds = matched[matched.get("home_no_vig_prob", pd.Series(np.nan, index=matched.index)).notna()]
        odds_counts = with_odds.groupby("season")["game_key"].nunique()
    rows = []
    for season_value, nba_games in game_counts.items():
        season = int(season_value)
        sportsbook_games = int(odds_counts.get(season_value, 0))
        rows.append(
            {
                "season": season,
                "season_label": nba_season_display_label(season),
                "nba_games": int(nba_games),
                "sportsbook_games": sportsbook_games,
                "sportsbook_match_rate": sportsbook_games / int(nba_games) if int(nba_games) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
