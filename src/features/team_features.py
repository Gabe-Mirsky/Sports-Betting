"""Assemble the leak-safe game-level modeling table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.cache import write_dataframe
from data.injury_availability import (
    AVAILABILITY_BASE_FEATURE_COLUMNS,
    build_game_availability_features,
    enrich_availability_reports_with_player_impact,
)
from data.seasons import assign_dataset_split
from data.validation import require_columns
from features.elo import add_elo_features, add_upcoming_elo_features
from features.player_features import PLAYER_BASE_FEATURE_COLUMNS, build_player_game_features
from features.rolling_stats import add_team_rolling_features, games_to_team_game_long
from features.schedule_features import add_schedule_features


ROLLING_WINDOWS = [3, 5, 10]


def _select_prefixed_team_features(
    team_games: pd.DataFrame,
    is_home: int,
    prefix: str,
) -> pd.DataFrame:
    feature_columns = [
        "rest_days",
        "is_back_to_back",
        "games_last_7_days",
        "games_last_14_days",
        "last_3_win_pct",
        "last_5_win_pct",
        "last_10_win_pct",
        "last_3_point_diff",
        "last_5_point_diff",
        "last_10_point_diff",
        "last_5_points_for",
        "last_5_points_against",
        "last_10_points_for",
        "last_10_points_against",
        "season_to_date_win_pct",
        "season_to_date_avg_margin",
        "season_to_date_points_for",
        "season_to_date_points_against",
        "last_5_fg_pct",
        "last_10_fg_pct",
        "last_5_fg3_pct",
        "last_10_fg3_pct",
        "last_5_ft_pct",
        "last_10_ft_pct",
        "last_5_reb",
        "last_10_reb",
        "last_5_ast",
        "last_10_ast",
        "last_5_tov",
        "last_10_tov",
    ]
    feature_columns = [column for column in feature_columns if column in team_games.columns]
    selected = team_games.loc[team_games["is_home"] == is_home, ["game_id", *feature_columns]].copy()
    rename_map = {column: f"{prefix}_{column}" for column in feature_columns}
    selected = selected.rename(columns=rename_map)
    selected = selected.rename(
        columns={
            f"{prefix}_season_to_date_win_pct": f"{prefix}_season_win_pct",
            f"{prefix}_season_to_date_avg_margin": f"{prefix}_season_avg_margin",
            f"{prefix}_season_to_date_points_for": f"{prefix}_season_points_for",
            f"{prefix}_season_to_date_points_against": f"{prefix}_season_points_against",
        }
    )
    return selected


def _add_difference_features(modeling: pd.DataFrame) -> pd.DataFrame:
    output = modeling.copy()
    output["rest_diff"] = output["home_rest_days"] - output["away_rest_days"]

    for window in ROLLING_WINDOWS:
        output[f"last_{window}_win_pct_diff"] = (
            output[f"home_last_{window}_win_pct"] - output[f"away_last_{window}_win_pct"]
        )
        output[f"last_{window}_point_diff_diff"] = (
            output[f"home_last_{window}_point_diff"]
            - output[f"away_last_{window}_point_diff"]
        )

    output["season_win_pct_diff"] = output["home_season_win_pct"] - output["away_season_win_pct"]
    output["season_avg_margin_diff"] = (
        output["home_season_avg_margin"] - output["away_season_avg_margin"]
    )
    for stat in ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov"]:
        for window in [5, 10]:
            home_column = f"home_last_{window}_{stat}"
            away_column = f"away_last_{window}_{stat}"
            if home_column in output.columns and away_column in output.columns:
                output[f"last_{window}_{stat}_diff"] = output[home_column] - output[away_column]
    for column in PLAYER_BASE_FEATURE_COLUMNS:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in output.columns and away_column in output.columns:
            output[f"{column}_diff"] = output[home_column] - output[away_column]
    for column in AVAILABILITY_BASE_FEATURE_COLUMNS:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in output.columns and away_column in output.columns:
            output[f"{column}_diff"] = output[home_column] - output[away_column]
    return output


def build_modeling_dataset(
    games: pd.DataFrame,
    player_logs: pd.DataFrame | None = None,
    availability_reports: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one modeling row per game with only pre-game features."""

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

    games_with_elo = add_elo_features(games)
    team_games = games_to_team_game_long(games)
    team_games = add_team_rolling_features(team_games, windows=ROLLING_WINDOWS)
    team_games = add_schedule_features(team_games)

    home_features = _select_prefixed_team_features(team_games, is_home=1, prefix="home")
    away_features = _select_prefixed_team_features(team_games, is_home=0, prefix="away")

    base_columns = [
        "game_id",
        "game_date",
        "season",
        "season_type",
        "home_team_id",
        "home_team_abbr",
        "away_team_id",
        "away_team_abbr",
        "neutral_site",
        "home_away_quality",
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff_pre",
        "elo_home_win_prob",
        "home_win",
    ]
    base_columns = [column for column in base_columns if column in games_with_elo.columns]
    modeling = games_with_elo[base_columns].copy()
    if "season_type" not in modeling.columns:
        modeling["season_type"] = "Regular Season"
    if "neutral_site" not in modeling.columns:
        modeling["neutral_site"] = 0
    modeling["neutral_site"] = pd.to_numeric(modeling["neutral_site"], errors="coerce").fillna(0).astype(int)
    modeling["is_playoffs"] = (
        modeling["season_type"].astype(str).str.lower().str.contains("playoff")
    ).astype(int)
    modeling = modeling.merge(home_features, on="game_id", how="left", validate="one_to_one")
    modeling = modeling.merge(away_features, on="game_id", how="left", validate="one_to_one")
    if player_logs is not None and not player_logs.empty:
        player_features = build_player_game_features(games, player_logs)
        modeling = modeling.merge(player_features, on="game_id", how="left", validate="one_to_one")
    if availability_reports is not None and not availability_reports.empty:
        if player_logs is not None and not player_logs.empty:
            availability_reports = enrich_availability_reports_with_player_impact(availability_reports, player_logs)
        availability_features = build_game_availability_features(games, availability_reports)
        if not availability_features.empty:
            modeling = modeling.merge(availability_features, on="game_id", how="left", validate="one_to_one")
    modeling["month"] = modeling["game_date"].dt.month
    modeling["day_of_week"] = modeling["game_date"].dt.dayofweek
    modeling["target_home_win"] = modeling["home_win"].astype(int)
    modeling["home_team_win"] = modeling["target_home_win"]
    modeling = _add_difference_features(modeling)
    modeling = assign_dataset_split(modeling)
    modeling = modeling.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return modeling


def build_upcoming_modeling_dataset(
    completed_games: pd.DataFrame,
    upcoming_games: pd.DataFrame,
    player_logs: pd.DataFrame | None = None,
    availability_reports: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build model-ready rows for scheduled games using only completed-game history."""

    if upcoming_games.empty:
        return pd.DataFrame()

    require_columns(
        completed_games,
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
        dataframe_name="completed_games",
    )
    require_columns(
        upcoming_games,
        [
            "game_id",
            "game_date",
            "season",
            "home_team_id",
            "home_team_abbr",
            "away_team_id",
            "away_team_abbr",
        ],
        dataframe_name="upcoming_games",
    )

    upcoming = upcoming_games.copy()
    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"], errors="coerce")
    if "season_type" not in upcoming.columns:
        upcoming["season_type"] = "Regular Season"
    if "upcoming_status" not in upcoming.columns:
        upcoming["upcoming_status"] = "Scheduled"

    history = completed_games.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    history = history[history["game_date"] < upcoming["game_date"].min()].copy()

    placeholder_columns = [
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
    ]
    for column in placeholder_columns:
        if column not in upcoming.columns:
            upcoming[column] = pd.NA

    combined_columns = sorted(set(history.columns).union(upcoming.columns))
    combined = pd.concat(
        [
            history.reindex(columns=combined_columns),
            upcoming.reindex(columns=combined_columns),
        ],
        ignore_index=True,
        sort=False,
    )

    team_games = games_to_team_game_long(combined)
    team_games = add_team_rolling_features(team_games, windows=ROLLING_WINDOWS)
    team_games = add_schedule_features(team_games)

    home_features = _select_prefixed_team_features(team_games, is_home=1, prefix="home")
    away_features = _select_prefixed_team_features(team_games, is_home=0, prefix="away")
    elo_features = add_upcoming_elo_features(history, upcoming)

    base_columns = [
        "game_id",
        "game_date",
        "season",
        "season_type",
        "home_team_id",
        "home_team_abbr",
        "away_team_id",
        "away_team_abbr",
        "neutral_site",
        "home_away_quality",
        "upcoming_status",
    ]
    if "game_status_id" in upcoming.columns:
        base_columns.append("game_status_id")
    modeling = upcoming[[column for column in base_columns if column in upcoming.columns]].copy()
    if "neutral_site" not in modeling.columns:
        modeling["neutral_site"] = 0
    modeling["neutral_site"] = pd.to_numeric(modeling["neutral_site"], errors="coerce").fillna(0).astype(int)
    modeling = modeling.merge(elo_features, on="game_id", how="left", validate="one_to_one")
    modeling = modeling.merge(home_features, on="game_id", how="left", validate="one_to_one")
    modeling = modeling.merge(away_features, on="game_id", how="left", validate="one_to_one")
    if player_logs is not None and not player_logs.empty:
        player_features = build_player_game_features(upcoming, player_logs)
        modeling = modeling.merge(player_features, on="game_id", how="left", validate="one_to_one")
    if availability_reports is not None and not availability_reports.empty:
        if player_logs is not None and not player_logs.empty:
            availability_reports = enrich_availability_reports_with_player_impact(availability_reports, player_logs)
        availability_features = build_game_availability_features(upcoming, availability_reports)
        if not availability_features.empty:
            modeling = modeling.merge(availability_features, on="game_id", how="left", validate="one_to_one")
    modeling["is_playoffs"] = (
        modeling["season_type"].astype(str).str.lower().str.contains("playoff")
    ).astype(int)
    modeling["month"] = modeling["game_date"].dt.month
    modeling["day_of_week"] = modeling["game_date"].dt.dayofweek
    modeling = _add_difference_features(modeling)
    return modeling.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def save_modeling_dataset(modeling: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the modeling dataset to parquet."""

    return write_dataframe(modeling, output_path)
