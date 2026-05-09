"""Leak-safe player rotation features from NBA player game logs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.validation import require_columns


PLAYER_BASE_FEATURE_COLUMNS = [
    "player_prior_games_last10",
    "player_top3_minutes_last10",
    "player_top5_minutes_last10",
    "player_top8_minutes_last10",
    "player_top8_points_last10",
    "player_top8_reb_last10",
    "player_top8_ast_last10",
    "player_top8_stock_last10",
    "player_top8_tov_last10",
    "player_top8_plus_minus_last10",
    "player_top8_value_last10",
    "player_top8_games_played_share_last10",
    "player_active_count_last5",
    "player_rotation_continuity_last5",
    "player_top_player_minutes_last10",
    "player_top_player_days_since_seen",
    "player_top3_available_last_game_share",
    "player_top8_available_last_game_share",
    "player_top3_minutes_last_game",
    "player_top8_minutes_last_game",
    "player_top8_value_last_game",
    "player_key_absence_minutes_last_game",
    "player_top8_minutes_gap_last_game",
]

PLAYER_DIFF_FEATURE_COLUMNS = [f"{column}_diff" for column in PLAYER_BASE_FEATURE_COLUMNS]


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _parse_minutes(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        try:
            minutes = float(parts[0])
            seconds = float(parts[1]) if len(parts) > 1 else 0.0
            return minutes + seconds / 60.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _extract_season(row: pd.Series) -> int | None:
    season_start_year = row.get("season_start_year")
    if pd.notna(season_start_year):
        return int(season_start_year)

    nba_season = row.get("nba_season")
    if isinstance(nba_season, str) and "-" in nba_season:
        return int(nba_season.split("-", maxsplit=1)[0])

    season_id = row.get("SEASON_ID")
    if pd.notna(season_id):
        text = str(int(season_id)) if isinstance(season_id, (int, float, np.integer, np.floating)) else str(season_id)
        return int(text[-4:])
    return None


def normalize_player_logs(player_logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize NBA player LeagueGameLog rows into stable numeric columns."""

    if player_logs.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "team_id",
                "team_abbr",
                "season",
                "player_id",
                "player_name",
                "minutes",
                "points",
                "rebounds",
                "assists",
                "steals",
                "blocks",
                "turnovers",
                "plus_minus",
                "box_score_value",
            ]
    )

    require_columns(
        player_logs,
        ["GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID"],
        dataframe_name="player_logs",
    )
    working = player_logs.copy()
    team_abbr = (
        working["TEAM_ABBREVIATION"]
        if "TEAM_ABBREVIATION" in working.columns
        else pd.Series("", index=working.index)
    )
    player_name = (
        working["PLAYER_NAME"]
        if "PLAYER_NAME" in working.columns
        else pd.Series("", index=working.index)
    )
    minutes = working["MIN"] if "MIN" in working.columns else pd.Series(0, index=working.index)
    output = pd.DataFrame(
        {
            "game_id": working["GAME_ID"].astype(str),
            "game_date": pd.to_datetime(working["GAME_DATE"], errors="coerce"),
            "team_id": pd.to_numeric(working["TEAM_ID"], errors="coerce"),
            "team_abbr": team_abbr.astype(str),
            "season": working.apply(_extract_season, axis=1),
            "player_id": working["PLAYER_ID"].astype(str),
            "player_name": player_name.astype(str),
            "minutes": minutes.map(_parse_minutes),
            "points": _numeric_column(working, "PTS"),
            "rebounds": _numeric_column(working, "REB"),
            "assists": _numeric_column(working, "AST"),
            "steals": _numeric_column(working, "STL"),
            "blocks": _numeric_column(working, "BLK"),
            "turnovers": _numeric_column(working, "TOV"),
            "plus_minus": _numeric_column(working, "PLUS_MINUS"),
        }
    )
    output["team_id"] = output["team_id"].astype("Int64")
    output["season"] = pd.to_numeric(output["season"], errors="coerce").astype("Int64")
    output["box_score_value"] = (
        output["points"]
        + 1.2 * output["rebounds"]
        + 1.5 * output["assists"]
        + 3.0 * (output["steals"] + output["blocks"])
        - output["turnovers"]
    )
    output = output.dropna(subset=["game_date", "team_id"]).copy()
    return output.sort_values(["team_id", "game_date", "game_id", "player_id"]).reset_index(drop=True)


def _games_to_team_targets(games: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        games,
        [
            "game_id",
            "game_date",
            "home_team_id",
            "home_team_abbr",
            "away_team_id",
            "away_team_abbr",
        ],
        dataframe_name="games",
    )
    working = games.copy()
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce")
    home = pd.DataFrame(
        {
            "game_id": working["game_id"].astype(str),
            "game_date": working["game_date"],
            "team_id": pd.to_numeric(working["home_team_id"], errors="coerce"),
            "team_abbr": working["home_team_abbr"].astype(str),
            "season": working["season"] if "season" in working.columns else pd.NA,
            "is_home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "game_id": working["game_id"].astype(str),
            "game_date": working["game_date"],
            "team_id": pd.to_numeric(working["away_team_id"], errors="coerce"),
            "team_abbr": working["away_team_abbr"].astype(str),
            "season": working["season"] if "season" in working.columns else pd.NA,
            "is_home": 0,
        }
    )
    targets = pd.concat([home, away], ignore_index=True)
    targets["team_id"] = targets["team_id"].astype("Int64")
    return targets.dropna(subset=["game_date", "team_id"]).sort_values(
        ["team_id", "game_date", "game_id", "is_home"]
    )


def _empty_feature_row() -> dict[str, float]:
    return {column: np.nan for column in PLAYER_BASE_FEATURE_COLUMNS}


def _feature_row_for_target(
    target: pd.Series,
    team_logs: pd.DataFrame,
    team_game_order: pd.DataFrame,
) -> dict[str, float]:
    current_date = pd.Timestamp(target["game_date"])
    prior_games = team_game_order[team_game_order["game_date"] < current_date].tail(10)
    if prior_games.empty:
        row = _empty_feature_row()
        row["player_prior_games_last10"] = 0.0
        return row

    last10_ids = set(prior_games["game_id"].astype(str))
    last5_ids = set(prior_games.tail(5)["game_id"].astype(str))
    last10_count = float(len(last10_ids))
    last5_count = float(len(last5_ids))
    recent = team_logs[team_logs["game_id"].astype(str).isin(last10_ids)].copy()
    if recent.empty:
        row = _empty_feature_row()
        row["player_prior_games_last10"] = last10_count
        return row

    grouped = recent.groupby(["player_id", "player_name"], dropna=False).agg(
        games_played=("game_id", "nunique"),
        last_seen=("game_date", "max"),
        minutes=("minutes", "sum"),
        points=("points", "sum"),
        rebounds=("rebounds", "sum"),
        assists=("assists", "sum"),
        steals=("steals", "sum"),
        blocks=("blocks", "sum"),
        turnovers=("turnovers", "sum"),
        plus_minus=("plus_minus", "sum"),
        box_score_value=("box_score_value", "sum"),
    )
    for column in [
        "minutes",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "plus_minus",
        "box_score_value",
    ]:
        grouped[f"{column}_per_team_game"] = grouped[column] / last10_count
    grouped["games_played_share"] = grouped["games_played"] / last10_count
    grouped = grouped.sort_values(
        ["minutes_per_team_game", "box_score_value_per_team_game", "games_played_share"],
        ascending=False,
    )

    top3 = grouped.head(3)
    top5 = grouped.head(5)
    top8 = grouped.head(8)
    top_player = grouped.iloc[0]

    last5 = team_logs[team_logs["game_id"].astype(str).isin(last5_ids)].copy()
    if last5.empty or last5_count <= 0:
        active_count_last5 = np.nan
        continuity_last5 = np.nan
    else:
        last5_grouped = last5.groupby("player_id").agg(
            games_played=("game_id", "nunique"),
            minutes=("minutes", "sum"),
        )
        last5_grouped["minutes_per_team_game"] = last5_grouped["minutes"] / last5_count
        active_count_last5 = float((last5_grouped["minutes_per_team_game"] >= 5.0).sum())
        top8_ids = set(top8.index.get_level_values("player_id"))
        if top8_ids:
            continuity_last5 = float(
                last5_grouped.reindex(list(top8_ids))["games_played"].fillna(0).sum()
                / (len(top8_ids) * last5_count)
            )
        else:
            continuity_last5 = np.nan

    stock = top8["steals_per_team_game"].sum() + top8["blocks_per_team_game"].sum()
    last_game_id = str(prior_games.tail(1)["game_id"].iloc[0])
    last_game = team_logs[team_logs["game_id"].astype(str).eq(last_game_id)].copy()
    last_game_grouped = last_game.groupby("player_id").agg(
        minutes=("minutes", "sum"),
        box_score_value=("box_score_value", "sum"),
    )
    top3_ids = list(top3.index.get_level_values("player_id"))
    top8_ids = list(top8.index.get_level_values("player_id"))
    top3_last = last_game_grouped.reindex(top3_ids).fillna(0)
    top8_last = last_game_grouped.reindex(top8_ids).fillna(0)
    top3_available_last_game_share = float((top3_last["minutes"] > 0).sum() / len(top3_ids)) if top3_ids else np.nan
    top8_available_last_game_share = float((top8_last["minutes"] > 0).sum() / len(top8_ids)) if top8_ids else np.nan
    expected_top8_minutes = top8["minutes_per_team_game"]
    expected_top8_minutes.index = expected_top8_minutes.index.get_level_values("player_id")
    missing_top8_ids = [player_id for player_id in top8_ids if player_id not in set(last_game_grouped.index)]
    key_absence_minutes = float(expected_top8_minutes.reindex(missing_top8_ids).fillna(0).sum())
    top8_minutes_last_game = float(top8_last["minutes"].sum())
    return {
        "player_prior_games_last10": last10_count,
        "player_top3_minutes_last10": float(top3["minutes_per_team_game"].sum()),
        "player_top5_minutes_last10": float(top5["minutes_per_team_game"].sum()),
        "player_top8_minutes_last10": float(top8["minutes_per_team_game"].sum()),
        "player_top8_points_last10": float(top8["points_per_team_game"].sum()),
        "player_top8_reb_last10": float(top8["rebounds_per_team_game"].sum()),
        "player_top8_ast_last10": float(top8["assists_per_team_game"].sum()),
        "player_top8_stock_last10": float(stock),
        "player_top8_tov_last10": float(top8["turnovers_per_team_game"].sum()),
        "player_top8_plus_minus_last10": float(top8["plus_minus_per_team_game"].sum()),
        "player_top8_value_last10": float(top8["box_score_value_per_team_game"].sum()),
        "player_top8_games_played_share_last10": float(top8["games_played_share"].mean()),
        "player_active_count_last5": active_count_last5,
        "player_rotation_continuity_last5": continuity_last5,
        "player_top_player_minutes_last10": float(top_player["minutes_per_team_game"]),
        "player_top_player_days_since_seen": float((current_date - top_player["last_seen"]).days),
        "player_top3_available_last_game_share": top3_available_last_game_share,
        "player_top8_available_last_game_share": top8_available_last_game_share,
        "player_top3_minutes_last_game": float(top3_last["minutes"].sum()),
        "player_top8_minutes_last_game": top8_minutes_last_game,
        "player_top8_value_last_game": float(top8_last["box_score_value"].sum()),
        "player_key_absence_minutes_last_game": key_absence_minutes,
        "player_top8_minutes_gap_last_game": float(expected_top8_minutes.sum() - top8_minutes_last_game),
    }


def build_team_player_features(games: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team-game with pregame player rotation features."""

    targets = _games_to_team_targets(games)
    normalized = normalize_player_logs(player_logs)
    base_columns = ["game_id", "team_id", "team_abbr", "is_home", *PLAYER_BASE_FEATURE_COLUMNS]
    if targets.empty or normalized.empty:
        output = targets[["game_id", "team_id", "team_abbr", "is_home"]].copy()
        for column in PLAYER_BASE_FEATURE_COLUMNS:
            output[column] = np.nan
        return output.reindex(columns=base_columns)

    rows: list[dict[str, Any]] = []
    team_logs_by_id = {team_id: group.copy() for team_id, group in normalized.groupby("team_id", sort=False)}
    for team_id, team_targets in targets.groupby("team_id", sort=False):
        team_logs = team_logs_by_id.get(team_id)
        if team_logs is None or team_logs.empty:
            for _, target in team_targets.iterrows():
                rows.append({**target.to_dict(), **_empty_feature_row()})
            continue
        for _, target in team_targets.iterrows():
            target_logs = team_logs
            if "season" in team_logs.columns and team_logs["season"].notna().any() and pd.notna(target.get("season")):
                target_logs = team_logs[team_logs["season"].eq(int(target["season"]))]
            team_game_order = (
                target_logs[["game_id", "game_date"]]
                .drop_duplicates()
                .sort_values(["game_date", "game_id"])
                .reset_index(drop=True)
            )
            rows.append(
                {
                    **target.to_dict(),
                    **_feature_row_for_target(target, target_logs, team_game_order),
                }
            )
    return pd.DataFrame(rows, columns=base_columns)


def build_player_game_features(games: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game with home/away player features and differences."""

    team_features = build_team_player_features(games, player_logs)
    if team_features.empty:
        return pd.DataFrame({"game_id": games["game_id"].astype(str)})

    home = team_features[team_features["is_home"].eq(1)][["game_id", *PLAYER_BASE_FEATURE_COLUMNS]].copy()
    away = team_features[team_features["is_home"].eq(0)][["game_id", *PLAYER_BASE_FEATURE_COLUMNS]].copy()
    home = home.rename(columns={column: f"home_{column}" for column in PLAYER_BASE_FEATURE_COLUMNS})
    away = away.rename(columns={column: f"away_{column}" for column in PLAYER_BASE_FEATURE_COLUMNS})
    output = pd.DataFrame({"game_id": games["game_id"].astype(str)})
    output = output.merge(home, on="game_id", how="left", validate="one_to_one")
    output = output.merge(away, on="game_id", how="left", validate="one_to_one")
    for column in PLAYER_BASE_FEATURE_COLUMNS:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in output.columns and away_column in output.columns:
            output[f"{column}_diff"] = output[home_column] - output[away_column]
    return output
