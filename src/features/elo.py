"""Simple pre-game Elo features for NBA games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.validation import require_columns


@dataclass(frozen=True)
class EloSettings:
    starting_elo: float = 1500.0
    k_factor: float = 20.0
    home_court_advantage: float = 60.0
    season_carryover: float = 0.75


def expected_home_win_prob(
    home_elo: float,
    away_elo: float,
    home_court_advantage: float = 60.0,
) -> float:
    """Return the Elo-implied home win probability."""

    adjusted_home_elo = home_elo + home_court_advantage
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home_elo) / 400.0))


def update_elo_ratings(
    home_elo: float,
    away_elo: float,
    actual_home_win: int,
    expected_home_win: float,
    k_factor: float = 20.0,
) -> tuple[float, float]:
    """Update home and away Elo ratings after a game."""

    change = k_factor * (actual_home_win - expected_home_win)
    return home_elo + change, away_elo - change


def _regress_to_mean(
    ratings: dict[Any, float],
    starting_elo: float,
    season_carryover: float,
) -> dict[Any, float]:
    return {
        team_id: season_carryover * rating + (1.0 - season_carryover) * starting_elo
        for team_id, rating in ratings.items()
    }


def add_elo_features(
    games: pd.DataFrame,
    settings: EloSettings | None = None,
) -> pd.DataFrame:
    """Add pre-game Elo features and update ratings after each game."""

    settings = settings or EloSettings()
    require_columns(
        games,
        [
            "game_id",
            "game_date",
            "season",
            "home_team_id",
            "away_team_id",
            "home_win",
        ],
        dataframe_name="games",
    )

    output = games.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output = output.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings: dict[Any, float] = {}
    current_season: int | None = None
    rows: list[dict[str, float]] = []

    for _, game in output.iterrows():
        season = int(game["season"])
        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = _regress_to_mean(
                ratings,
                settings.starting_elo,
                settings.season_carryover,
            )
            current_season = season

        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]
        home_elo_pre = ratings.get(home_team_id, settings.starting_elo)
        away_elo_pre = ratings.get(away_team_id, settings.starting_elo)
        expected = expected_home_win_prob(
            home_elo_pre,
            away_elo_pre,
            settings.home_court_advantage,
        )

        rows.append(
            {
                "home_elo_pre": home_elo_pre,
                "away_elo_pre": away_elo_pre,
                "elo_diff_pre": home_elo_pre - away_elo_pre,
                "elo_home_win_prob": expected,
            }
        )

        home_elo_post, away_elo_post = update_elo_ratings(
            home_elo_pre,
            away_elo_pre,
            int(game["home_win"]),
            expected,
            settings.k_factor,
        )
        ratings[home_team_id] = home_elo_post
        ratings[away_team_id] = away_elo_post

    elo_features = pd.DataFrame(rows)
    return pd.concat([output, elo_features], axis=1)


def current_elo_state(
    games: pd.DataFrame,
    settings: EloSettings | None = None,
) -> tuple[dict[Any, float], int | None]:
    """Return Elo ratings after the latest completed game."""

    settings = settings or EloSettings()
    require_columns(
        games,
        [
            "game_id",
            "game_date",
            "season",
            "home_team_id",
            "away_team_id",
            "home_win",
        ],
        dataframe_name="games",
    )

    completed = games.dropna(subset=["home_win"]).copy()
    if completed.empty:
        return {}, None

    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed = completed.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings: dict[Any, float] = {}
    current_season: int | None = None
    for _, game in completed.iterrows():
        season = int(game["season"])
        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = _regress_to_mean(
                ratings,
                settings.starting_elo,
                settings.season_carryover,
            )
            current_season = season

        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]
        home_elo_pre = ratings.get(home_team_id, settings.starting_elo)
        away_elo_pre = ratings.get(away_team_id, settings.starting_elo)
        expected = expected_home_win_prob(
            home_elo_pre,
            away_elo_pre,
            settings.home_court_advantage,
        )
        home_elo_post, away_elo_post = update_elo_ratings(
            home_elo_pre,
            away_elo_pre,
            int(game["home_win"]),
            expected,
            settings.k_factor,
        )
        ratings[home_team_id] = home_elo_post
        ratings[away_team_id] = away_elo_post

    return ratings, current_season


def add_upcoming_elo_features(
    completed_games: pd.DataFrame,
    upcoming_games: pd.DataFrame,
    settings: EloSettings | None = None,
) -> pd.DataFrame:
    """Add current pre-game Elo features for scheduled future games."""

    settings = settings or EloSettings()
    require_columns(
        upcoming_games,
        [
            "game_id",
            "game_date",
            "season",
            "home_team_id",
            "away_team_id",
        ],
        dataframe_name="upcoming_games",
    )

    ratings, current_season = current_elo_state(completed_games, settings=settings)
    upcoming = upcoming_games.copy()
    upcoming["game_date"] = pd.to_datetime(upcoming["game_date"], errors="coerce")
    upcoming = upcoming.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    rows: list[dict[str, float | str]] = []
    elo_season = current_season
    for _, game in upcoming.iterrows():
        season = int(game["season"])
        if elo_season is not None and season != elo_season:
            ratings = _regress_to_mean(
                ratings,
                settings.starting_elo,
                settings.season_carryover,
            )
            elo_season = season
        elif elo_season is None:
            elo_season = season

        home_team_id = game["home_team_id"]
        away_team_id = game["away_team_id"]
        home_elo_pre = ratings.get(home_team_id, settings.starting_elo)
        away_elo_pre = ratings.get(away_team_id, settings.starting_elo)
        expected = expected_home_win_prob(
            home_elo_pre,
            away_elo_pre,
            settings.home_court_advantage,
        )
        rows.append(
            {
                "game_id": game["game_id"],
                "home_elo_pre": home_elo_pre,
                "away_elo_pre": away_elo_pre,
                "elo_diff_pre": home_elo_pre - away_elo_pre,
                "elo_home_win_prob": expected,
            }
        )

    return pd.DataFrame(rows)
