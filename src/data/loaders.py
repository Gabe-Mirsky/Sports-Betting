"""Load and reshape cached NBA data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import read_dataframe, write_dataframe
from .validation import require_columns


logger = logging.getLogger(__name__)


GAME_LEVEL_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "season_type",
    "home_team_id",
    "home_team_abbr",
    "away_team_id",
    "away_team_abbr",
    "home_points",
    "away_points",
    "home_win",
    "away_win",
    "home_plus_minus",
    "away_plus_minus",
    "home_fg_pct",
    "away_fg_pct",
    "home_fg3_pct",
    "away_fg3_pct",
    "home_ft_pct",
    "away_ft_pct",
    "home_reb",
    "away_reb",
    "home_ast",
    "away_ast",
    "home_tov",
    "away_tov",
    "neutral_site",
    "home_away_quality",
]

BOX_SCORE_STAT_COLUMNS = {
    "fg_pct": "FG_PCT",
    "fg3_pct": "FG3_PCT",
    "ft_pct": "FT_PCT",
    "reb": "REB",
    "ast": "AST",
    "tov": "TOV",
}


def load_raw_team_logs(raw_dir: str | Path) -> pd.DataFrame:
    """Load all cached raw NBA LeagueGameLog parquet files."""

    raw_path = Path(raw_dir)
    files = sorted(
        {
            *raw_path.glob("league_game_log_*.parquet"),
            *raw_path.glob("nba_league_game_log_*.parquet"),
        },
        key=lambda path: path.name,
    )
    if not files and raw_path.name.lower() == "nba":
        legacy_path = raw_path.parent
        files = sorted(legacy_path.glob("nba_league_game_log_*.parquet"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(
            f"No NBA LeagueGameLog parquet files found in {raw_path}. "
            "Run scripts/download_nba_data.py first."
        )

    frames = []
    for file_path in files:
        frame = read_dataframe(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    output = pd.concat(frames, ignore_index=True)
    if "GAME_DATE" in output.columns:
        output["GAME_DATE"] = pd.to_datetime(output["GAME_DATE"], errors="coerce")

    logger.info("Loaded %s team-game rows from %s cached files", len(output), len(files))
    return output


def _extract_season(row: pd.Series) -> int | None:
    season_start_year = row.get("season_start_year")
    if pd.notna(season_start_year):
        return int(season_start_year)

    nba_season = row.get("nba_season")
    if isinstance(nba_season, str) and "-" in nba_season:
        return int(nba_season.split("-", maxsplit=1)[0])

    season_id = row.get("SEASON_ID")
    if pd.notna(season_id):
        if isinstance(season_id, (int, float)):
            season_id_text = str(int(season_id))
        else:
            season_id_text = str(season_id)
        return int(season_id_text[-4:])

    return None


def _is_home_matchup(matchup: Any) -> bool:
    matchup_text = str(matchup)
    return " vs. " in matchup_text or "vs." in matchup_text


def _opponent_from_matchup(matchup: Any) -> str | None:
    matchup_text = str(matchup)
    separator = " vs. " if " vs. " in matchup_text else " @ " if " @ " in matchup_text else None
    if separator is None:
        return None
    parts = matchup_text.split(separator, maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def _infer_neutral_site_home_away(group: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Infer a designated side for neutral-site games with two away-style rows.

    NBA's LeagueGameLog sometimes marks both team rows with '@' for neutral-site
    and cup games. We keep those rows, mark them as inferred neutral-site games,
    and use the first schedule row's opponent as the designated home side.
    """

    away_style = group[group["MATCHUP"].astype(str).str.contains(" @ ", regex=False)]
    if len(away_style) != 2:
        return None

    first = away_style.sort_index().iloc[0]
    opponent_abbr = _opponent_from_matchup(first.get("MATCHUP"))
    if not opponent_abbr:
        return None
    home_group = group[group["TEAM_ABBREVIATION"].astype(str).eq(opponent_abbr)]
    away_group = group[group.index == first.name]
    if len(home_group) != 1 or len(away_group) != 1:
        return None
    return home_group.iloc[0], away_group.iloc[0]


def _safe_numeric(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def build_game_level_dataset(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Collapse team-level game logs into one row per NBA game."""

    require_columns(
        team_logs,
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "MATCHUP",
            "PTS",
            "PLUS_MINUS",
        ],
        dataframe_name="team_logs",
    )

    working = team_logs.copy()
    working["GAME_DATE"] = pd.to_datetime(working["GAME_DATE"], errors="coerce")
    working["PTS"] = pd.to_numeric(working["PTS"], errors="coerce")
    working["PLUS_MINUS"] = pd.to_numeric(working["PLUS_MINUS"], errors="coerce")

    rows: list[dict[str, Any]] = []
    invalid_games = 0

    for game_id, group in working.groupby("GAME_ID", sort=False):
        if len(group) != 2:
            invalid_games += 1
            logger.warning("Skipping GAME_ID=%s because it has %s rows", game_id, len(group))
            continue

        neutral_site = 0
        home_away_quality = "standard_matchup"
        home_group = group[group["MATCHUP"].apply(_is_home_matchup)]
        if len(home_group) == 1:
            home = home_group.iloc[0]
            away = group[group.index != home.name].iloc[0]
        else:
            inferred = _infer_neutral_site_home_away(group)
            if inferred is None:
                invalid_games += 1
                logger.warning("Skipping GAME_ID=%s because home team parsing failed", game_id)
                continue
            home, away = inferred
            neutral_site = 1
            home_away_quality = "neutral_site_inferred"
            logger.info("Inferred neutral-site home/away for GAME_ID=%s", game_id)

        home_points = _safe_numeric(home["PTS"])
        away_points = _safe_numeric(away["PTS"])
        home_plus_minus = _safe_numeric(home["PLUS_MINUS"])
        away_plus_minus = _safe_numeric(away["PLUS_MINUS"])
        if home_points is None or away_points is None:
            invalid_games += 1
            logger.warning("Skipping GAME_ID=%s because score parsing failed", game_id)
            continue

        home_box_score_stats = {
            f"home_{name}": _safe_numeric(home.get(raw_column))
            for name, raw_column in BOX_SCORE_STAT_COLUMNS.items()
        }
        away_box_score_stats = {
            f"away_{name}": _safe_numeric(away.get(raw_column))
            for name, raw_column in BOX_SCORE_STAT_COLUMNS.items()
        }

        home_win = int(home_points > away_points)
        rows.append(
            {
                "game_id": str(game_id),
                "game_date": home["GAME_DATE"],
                "season": _extract_season(home),
                "season_type": str(home.get("season_type", "Regular Season")),
                "home_team_id": int(home["TEAM_ID"]),
                "home_team_abbr": str(home["TEAM_ABBREVIATION"]),
                "away_team_id": int(away["TEAM_ID"]),
                "away_team_abbr": str(away["TEAM_ABBREVIATION"]),
                "home_points": home_points,
                "away_points": away_points,
                "home_win": home_win,
                "away_win": int(not home_win),
                "home_plus_minus": home_plus_minus,
                "away_plus_minus": away_plus_minus,
                **home_box_score_stats,
                **away_box_score_stats,
                "neutral_site": neutral_site,
                "home_away_quality": home_away_quality,
            }
        )

    games = pd.DataFrame(rows, columns=GAME_LEVEL_COLUMNS)
    if games.empty:
        logger.warning("Built an empty game-level dataset")
        return games

    duplicate_count = int(games["game_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Game-level dataset has {duplicate_count} duplicate GAME_ID rows")

    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    logger.info("Built %s game rows; skipped %s invalid games", len(games), invalid_games)
    return games


def save_game_level_dataset(games: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the game-level dataset to parquet."""

    return write_dataframe(games, output_path)


def load_game_level_dataset(path: str | Path) -> pd.DataFrame:
    """Load a saved game-level parquet dataset."""

    games = read_dataframe(path)
    if "game_date" in games.columns:
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    return games
