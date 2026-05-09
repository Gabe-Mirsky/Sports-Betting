"""Leak-safe rolling team features."""

from __future__ import annotations

import pandas as pd

from data.validation import require_columns


def games_to_team_game_long(games: pd.DataFrame) -> pd.DataFrame:
    """Convert one-row-per-game data into one row per team per game."""

    require_columns(
        games,
        [
            "game_id",
            "game_date",
            "season",
            "home_team_id",
            "home_team_abbr",
            "away_team_id",
            "away_team_abbr",
            "home_points",
            "away_points",
            "home_win",
        ],
        dataframe_name="games",
    )

    working = games.copy()
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce")

    home_data = {
        "game_id": working["game_id"],
        "game_date": working["game_date"],
        "season": working["season"],
        "season_type": working["season_type"] if "season_type" in working.columns else "Regular Season",
        "team_id": working["home_team_id"],
        "team_abbr": working["home_team_abbr"],
        "opponent_team_id": working["away_team_id"],
        "opponent_team_abbr": working["away_team_abbr"],
        "is_home": 1,
        "points_for": working["home_points"],
        "points_against": working["away_points"],
        "win": working["home_win"],
    }
    away_data = {
        "game_id": working["game_id"],
        "game_date": working["game_date"],
        "season": working["season"],
        "season_type": working["season_type"] if "season_type" in working.columns else "Regular Season",
        "team_id": working["away_team_id"],
        "team_abbr": working["away_team_abbr"],
        "opponent_team_id": working["home_team_id"],
        "opponent_team_abbr": working["home_team_abbr"],
        "is_home": 0,
        "points_for": working["away_points"],
        "points_against": working["home_points"],
        "win": working["away_win"] if "away_win" in working.columns else 1 - working["home_win"],
    }

    optional_stats = ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]
    for stat in optional_stats:
        home_column = f"home_{stat}"
        away_column = f"away_{stat}"
        if home_column in working.columns:
            home_data[stat] = working[home_column]
        if away_column in working.columns:
            away_data[stat] = working[away_column]

    home = pd.DataFrame(home_data)
    away = pd.DataFrame(away_data)

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["margin"] = team_games["points_for"] - team_games["points_against"]
    team_games["home_game"] = team_games["is_home"].astype(int)
    team_games["away_game"] = 1 - team_games["home_game"]
    team_games = team_games.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    return team_games


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shifted_expanding_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def add_team_rolling_features(
    team_games: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling and season-to-date features using only previous games."""

    windows = windows or [3, 5, 10]
    require_columns(
        team_games,
        [
            "team_id",
            "game_date",
            "season",
            "win",
            "margin",
            "points_for",
            "points_against",
        ],
        dataframe_name="team_games",
    )

    output = team_games.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output = output.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    for column in ["win", "margin", "points_for", "points_against"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    by_team = output.groupby("team_id", sort=False)

    for window in windows:
        output[f"last_{window}_win_pct"] = by_team["win"].transform(
            lambda series: _shifted_rolling_mean(series, window)
        )
        output[f"last_{window}_point_diff"] = by_team["margin"].transform(
            lambda series: _shifted_rolling_mean(series, window)
        )
        output[f"last_{window}_points_for"] = by_team["points_for"].transform(
            lambda series: _shifted_rolling_mean(series, window)
        )
        output[f"last_{window}_points_against"] = by_team["points_against"].transform(
            lambda series: _shifted_rolling_mean(series, window)
        )

    optional_rolling_stats = [stat for stat in ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"] if stat in output.columns]
    for stat in optional_rolling_stats:
        numeric = pd.to_numeric(output[stat], errors="coerce")
        output[stat] = numeric
        output[f"last_5_{stat}"] = by_team[stat].transform(
            lambda series: _shifted_rolling_mean(series, 5)
        )
        output[f"last_10_{stat}"] = by_team[stat].transform(
            lambda series: _shifted_rolling_mean(series, 10)
        )

    by_team_season = output.groupby(["team_id", "season"], sort=False)
    output["season_to_date_win_pct"] = by_team_season["win"].transform(_shifted_expanding_mean)
    output["season_to_date_avg_margin"] = by_team_season["margin"].transform(_shifted_expanding_mean)
    output["season_to_date_points_for"] = by_team_season["points_for"].transform(
        _shifted_expanding_mean
    )
    output["season_to_date_points_against"] = by_team_season["points_against"].transform(
        _shifted_expanding_mean
    )

    return output
