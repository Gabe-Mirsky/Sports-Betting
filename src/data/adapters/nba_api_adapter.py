"""nba_api adapter: player/team actuals, schedules, and prop settlement values.

Wraps the project's existing ``player_client`` cache so it works offline from
already-downloaded parquet/CSV. It supplies no betting markets - only the actuals
that settle props and feed features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..source_adapter import SportsDataSourceAdapter


# nba_api LeagueGameLog (player) column -> canonical player_game_log column.
_PLAYER_LOG_RENAME = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ABBREVIATION": "team_abbr",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "MIN": "minutes",
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "FG3M": "threes",
    "BLK": "blocks",
    "STL": "steals",
    "TOV": "turnovers",
    "season_start_year": "season",
}


class NbaApiAdapter(SportsDataSourceAdapter):
    """Actuals/feature source backed by nba_api caches."""

    source_key = "nba_api"
    sport = "basketball"
    league = "NBA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.cache_dir = Path(self.config.get("player_cache_dir", "data/raw/nba/player"))

    def fetch_player_game_logs(
        self,
        season: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        player_id: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load cached player game logs (no network). Normalize before returning."""

        from ..player_client import load_raw_player_logs

        raw = load_raw_player_logs(self.cache_dir)
        if raw.empty:
            return self.normalize_to_project_schema(pd.DataFrame(), "player_game_logs")
        if season is not None and "season_start_year" in raw.columns:
            raw = raw[pd.to_numeric(raw["season_start_year"], errors="coerce").eq(int(season))]
        if player_id is not None and "PLAYER_ID" in raw.columns:
            raw = raw[raw["PLAYER_ID"].astype(str).eq(str(player_id))]
        return self.normalize_to_project_schema(raw, "player_game_logs")

    def fetch_results(
        self,
        season: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Settlement values are the box-score actuals themselves (long form)."""

        logs = self.fetch_player_game_logs(season=season, start_date=start_date, end_date=end_date)
        if logs.empty:
            return self.normalize_to_project_schema(pd.DataFrame(), "results")
        stat_columns = ["points", "rebounds", "assists", "threes", "blocks", "steals", "turnovers"]
        melted = logs.melt(
            id_vars=["player_id", "player_name", "game_id", "game_date", "season"],
            value_vars=[column for column in stat_columns if column in logs.columns],
            var_name="stat_type",
            value_name="actual_value",
        )
        melted["settlement_price"] = pd.NA
        melted["outcome"] = pd.NA
        return self.normalize_to_project_schema(melted, "results")

    def normalize_to_project_schema(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return super().normalize_to_project_schema(pd.DataFrame(), entity)
        if entity == "player_game_logs":
            frame = self._normalize_player_logs(frame)
        return super().normalize_to_project_schema(frame, entity)

    def _normalize_player_logs(self, frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.rename(columns=_PLAYER_LOG_RENAME).copy()
        if "game_date" in renamed.columns:
            renamed["game_date"] = pd.to_datetime(renamed["game_date"], errors="coerce").dt.normalize()
        matchup = frame.get("MATCHUP")
        if matchup is not None:
            text = matchup.astype(str)
            renamed["is_home"] = text.str.contains("vs", case=False, regex=False)
            renamed["opponent_abbr"] = text.str.split().str[-1]
        if "season" not in renamed.columns and "nba_season" in frame.columns:
            renamed["season"] = (
                frame["nba_season"].astype(str).str.slice(0, 4).pipe(pd.to_numeric, errors="coerce")
            )
        return renamed
