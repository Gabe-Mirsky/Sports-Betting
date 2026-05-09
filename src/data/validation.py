"""Validation helpers for NBA and market dataframes."""

from __future__ import annotations

import pandas as pd


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str = "dataframe",
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataframe_name} is missing required columns: {missing}")


def validate_game_level_dataset(games: pd.DataFrame) -> list[str]:
    """Return validation issues for a one-row-per-game dataset."""

    issues: list[str] = []
    require_columns(
        games,
        [
            "game_id",
            "game_date",
            "home_team_abbr",
            "away_team_abbr",
            "home_points",
            "away_points",
            "home_win",
        ],
        dataframe_name="games",
    )

    if games["game_id"].duplicated().any():
        issues.append("Duplicate game_id rows found.")

    if games["game_date"].isna().any():
        issues.append("Some game_date values could not be parsed.")

    expected_home_win = (games["home_points"] > games["away_points"]).astype(int)
    if not (games["home_win"].astype(int) == expected_home_win).all():
        issues.append("home_win does not match home_points > away_points.")

    return issues
