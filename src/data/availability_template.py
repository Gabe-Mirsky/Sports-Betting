"""Build injury/availability entry templates from cached NBA player logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.team_aliases import normalize_team_abbr
from data.validation import require_columns
from features.player_features import normalize_player_logs


AVAILABILITY_TEMPLATE_COLUMNS = [
    "report_date",
    "game_date",
    "game_id",
    "team_abbr",
    "opponent_abbr",
    "home_away",
    "player_id",
    "player_name",
    "status",
    "impact_weight",
    "impact_weight_source",
    "impact_prior_games",
    "impact_avg_box_score_value_last10",
]


def _team_targets(games: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        games,
        ["game_id", "game_date", "home_team_abbr", "away_team_abbr"],
        dataframe_name="games",
    )
    working = games.copy()
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce")
    home = pd.DataFrame(
        {
            "game_id": working["game_id"].astype(str),
            "game_date": working["game_date"],
            "team_abbr": working["home_team_abbr"].map(normalize_team_abbr),
            "opponent_abbr": working["away_team_abbr"].map(normalize_team_abbr),
            "home_away": "home",
        }
    )
    away = pd.DataFrame(
        {
            "game_id": working["game_id"].astype(str),
            "game_date": working["game_date"],
            "team_abbr": working["away_team_abbr"].map(normalize_team_abbr),
            "opponent_abbr": working["home_team_abbr"].map(normalize_team_abbr),
            "home_away": "away",
        }
    )
    return (
        pd.concat([home, away], ignore_index=True)
        .dropna(subset=["game_date"])
        .sort_values(["game_date", "game_id", "home_away"])
        .reset_index(drop=True)
    )


def _top_players_for_team_game(
    team_logs: pd.DataFrame,
    game_date: pd.Timestamp,
    lookback_games: int,
    players_per_team: int,
) -> list[dict[str, Any]]:
    if team_logs.empty or "game_date" not in team_logs.columns:
        return []
    prior_games = (
        team_logs.loc[team_logs["game_date"] < game_date, ["game_id", "game_date"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"])
        .tail(lookback_games)
    )
    if prior_games.empty:
        return []
    recent = team_logs[team_logs["game_id"].astype(str).isin(prior_games["game_id"].astype(str))].copy()
    if recent.empty:
        return []
    denominator = float(len(prior_games))
    grouped = (
        recent.groupby(["player_id", "player_name"], dropna=False)
        .agg(
            games_played=("game_id", "nunique"),
            minutes=("minutes", "sum"),
            box_score_value=("box_score_value", "sum"),
        )
        .reset_index()
    )
    grouped["impact_weight"] = grouped["minutes"] / denominator
    grouped["impact_avg_box_score_value_last10"] = grouped["box_score_value"] / denominator
    grouped = grouped.sort_values(
        ["impact_weight", "impact_avg_box_score_value_last10", "games_played"],
        ascending=False,
    ).head(players_per_team)
    rows = []
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "player_id": str(row.player_id),
                "player_name": str(row.player_name),
                "impact_weight": round(float(row.impact_weight), 3),
                "impact_weight_source": f"player_logs_last_{lookback_games}_games_minutes",
                "impact_prior_games": int(row.games_played),
                "impact_avg_box_score_value_last10": round(float(row.impact_avg_box_score_value_last10), 3),
            }
        )
    return rows


def build_availability_template(
    games: pd.DataFrame,
    player_logs: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    players_per_team: int = 10,
    lookback_games: int = 10,
) -> pd.DataFrame:
    """Create a status-entry template with recent player impact weights."""

    if games.empty:
        return pd.DataFrame(columns=AVAILABILITY_TEMPLATE_COLUMNS)
    targets = _team_targets(games)
    if start_date is not None:
        targets = targets[targets["game_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        targets = targets[targets["game_date"] <= pd.Timestamp(end_date)]
    if targets.empty or player_logs.empty:
        return pd.DataFrame(columns=AVAILABILITY_TEMPLATE_COLUMNS)

    logs = normalize_player_logs(player_logs)
    if logs.empty:
        return pd.DataFrame(columns=AVAILABILITY_TEMPLATE_COLUMNS)
    logs["team_abbr"] = logs["team_abbr"].map(normalize_team_abbr)
    logs_by_team = {team: frame.copy() for team, frame in logs.groupby("team_abbr", sort=False)}

    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        team_logs = logs_by_team.get(target.team_abbr, pd.DataFrame())
        top_players = _top_players_for_team_game(
            team_logs,
            pd.Timestamp(target.game_date),
            lookback_games=lookback_games,
            players_per_team=players_per_team,
        )
        for player in top_players:
            rows.append(
                {
                    "report_date": pd.Timestamp(target.game_date).date().isoformat(),
                    "game_date": pd.Timestamp(target.game_date).date().isoformat(),
                    "game_id": target.game_id,
                    "team_abbr": target.team_abbr,
                    "opponent_abbr": target.opponent_abbr,
                    "home_away": target.home_away,
                    **player,
                    "status": "",
                }
            )
    return pd.DataFrame(rows, columns=AVAILABILITY_TEMPLATE_COLUMNS)


def write_availability_template(template: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return path
