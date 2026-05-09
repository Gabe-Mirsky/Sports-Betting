"""Target-building helpers."""

from __future__ import annotations

import pandas as pd

from data.validation import require_columns


def add_home_win_target(games: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical home-win target columns."""

    require_columns(games, ["home_points", "away_points"], dataframe_name="games")
    output = games.copy()
    output["target_home_win"] = (output["home_points"] > output["away_points"]).astype(int)
    output["home_team_win"] = output["target_home_win"]
    return output
