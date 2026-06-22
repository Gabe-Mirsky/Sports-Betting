"""Basketball-Reference adapter: fallback box-score actuals (manual import).

No betting odds. Basketball-Reference's ToS discourages bulk scraping and rate
limits aggressively (~20 req/min), so this adapter is built for *manual* exports:
point it at a downloaded CSV and it normalizes. Live scraping is intentionally
not implemented; use it only as a fallback when nba_api is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..source_adapter import SportsDataSourceAdapter
from .kaggle_csv_adapter import BOX_SCORE_ALIASES, apply_aliases


class BasketballReferenceAdapter(SportsDataSourceAdapter):
    """Manual-import fallback for player box-score actuals."""

    source_key = "basketball_reference"
    sport = "basketball"
    league = "NBA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.export_path = self.config.get("export_path")

    def fetch_player_game_logs(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, player_id: str | None = None, **kwargs: Any) -> pd.DataFrame:
        path = kwargs.get("path", self.export_path)
        if not path:
            raise RuntimeError(
                "BasketballReferenceAdapter is manual-import only (no live scraping). Download a game-log "
                "CSV and pass path=... or config={'export_path': ...}. Respect the site's ToS and rate limits."
            )
        frame = pd.read_csv(Path(path), low_memory=False)
        return self.normalize_to_project_schema(frame, "player_game_logs")

    def fetch_results(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, **kwargs: Any) -> pd.DataFrame:
        logs = self.fetch_player_game_logs(season=season, **kwargs)
        if logs.empty:
            return self.normalize_to_project_schema(pd.DataFrame(), "results")
        stat_columns = [c for c in ["points", "rebounds", "assists", "threes", "blocks", "steals", "turnovers"] if c in logs.columns]
        melted = logs.melt(
            id_vars=[c for c in ["player_id", "player_name", "game_id", "game_date", "season"] if c in logs.columns],
            value_vars=stat_columns,
            var_name="stat_type",
            value_name="actual_value",
        )
        return self.normalize_to_project_schema(melted, "results")

    def normalize_to_project_schema(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return super().normalize_to_project_schema(pd.DataFrame(), entity)
        if entity in {"player_game_logs", "results"}:
            frame = apply_aliases(frame, BOX_SCORE_ALIASES)
            if "game_date" in frame.columns:
                frame = frame.copy()
                frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
        return super().normalize_to_project_schema(frame, entity)
