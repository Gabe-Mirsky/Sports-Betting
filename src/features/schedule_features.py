"""Schedule and rest features built from past games only."""

from __future__ import annotations

import pandas as pd

from data.validation import require_columns


def _count_recent_games(dates: pd.Series, days: int) -> pd.Series:
    counts: list[int] = []
    previous_dates: list[pd.Timestamp] = []

    for current_date in dates:
        current_timestamp = pd.Timestamp(current_date)
        counts.append(
            sum(
                0 <= (current_timestamp - previous_date).days <= days
                for previous_date in previous_dates
            )
        )
        previous_dates.append(current_timestamp)

    return pd.Series(counts, index=dates.index)


def add_schedule_features(team_game_long: pd.DataFrame) -> pd.DataFrame:
    """Add rest-day and recent-schedule features for each team-game row."""

    require_columns(
        team_game_long,
        ["team_id", "game_date", "season", "is_home"],
        dataframe_name="team_game_long",
    )

    output = team_game_long.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output = output.sort_values(["team_id", "season", "game_date", "game_id"]).reset_index(drop=True)

    by_team_season = output.groupby(["team_id", "season"], sort=False)
    previous_game_date = by_team_season["game_date"].shift(1)
    output["rest_days"] = (output["game_date"] - previous_game_date).dt.days
    output["rest_days"] = output["rest_days"].fillna(7).clip(lower=0)
    output["is_back_to_back"] = (output["rest_days"] <= 1).astype(int)
    output["home_game"] = output["is_home"].astype(int)
    output["away_game"] = 1 - output["home_game"]

    output["games_last_7_days"] = by_team_season["game_date"].transform(
        lambda dates: _count_recent_games(dates, 7)
    )
    output["games_last_14_days"] = by_team_season["game_date"].transform(
        lambda dates: _count_recent_games(dates, 14)
    )
    output["month"] = output["game_date"].dt.month
    output["day_of_week"] = output["game_date"].dt.dayofweek

    return output
