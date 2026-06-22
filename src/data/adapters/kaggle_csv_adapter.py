"""Kaggle / manual CSV adapter: historical box scores and team-market odds.

Reads static CSV exports and maps inconsistent third-party headers onto the
canonical tables via case-insensitive aliases. Most NBA datasets carry box-score
actuals and/or team moneyline/spread/total odds; historical *player-prop* odds
are rare, so prop pricing is not assumed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..source_adapter import SportsDataSourceAdapter


# Canonical column -> accepted source header aliases (compared lower-cased).
BOX_SCORE_ALIASES: dict[str, tuple[str, ...]] = {
    "player_id": ("player_id", "bbref_id", "slug", "playerid"),
    "player_name": ("player_name", "player", "name", "athlete"),
    "team_abbr": ("team_abbr", "team", "tm", "team_abbreviation"),
    "opponent_abbr": ("opponent_abbr", "opponent", "opp", "opp_abbr"),
    "game_id": ("game_id", "gid", "gameid"),
    "game_date": ("game_date", "date", "gamedate"),
    "season": ("season", "year", "season_start_year"),
    "minutes": ("minutes", "min", "mp"),
    "points": ("points", "pts"),
    "rebounds": ("rebounds", "reb", "trb"),
    "assists": ("assists", "ast"),
    "threes": ("threes", "fg3m", "3pm", "three_pointers_made"),
    "blocks": ("blocks", "blk"),
    "steals": ("steals", "stl"),
    "turnovers": ("turnovers", "tov", "to"),
}


def apply_aliases(frame: pd.DataFrame, alias_map: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Rename source columns to canonical names using case-insensitive aliases."""

    lookup = {str(column).lower(): column for column in frame.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias.lower() in lookup:
                rename[lookup[alias.lower()]] = canonical
                break
    return frame.rename(columns=rename)


class KaggleCsvAdapter(SportsDataSourceAdapter):
    """Static CSV import for historical actuals and team-market odds."""

    source_key = "kaggle_csv"
    sport = "basketball"
    league = "NBA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.box_score_path = self.config.get("box_score_path")
        self.odds_path = self.config.get("odds_path", "data/raw/sportsbook/kaggle")

    def fetch_player_game_logs(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, player_id: str | None = None, **kwargs: Any) -> pd.DataFrame:
        path = kwargs.get("path", self.box_score_path)
        if not path:
            raise RuntimeError(
                "KaggleCsvAdapter.fetch_player_game_logs needs a CSV path. Pass config={'box_score_path': ...} "
                "or path=... pointing at a downloaded Kaggle box-score export."
            )
        frame = _read_any(Path(path))
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

    def fetch_market_odds(self, event_id: str | None = None, market_types: list[str] | None = None, snapshot_time: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Load team-market odds via the existing sportsbook loader (moneyline/spread/total)."""

        from ..sportsbook_odds import load_sportsbook_odds

        path = kwargs.get("path", self.config.get("odds_file"))
        if not path:
            raise RuntimeError(
                "KaggleCsvAdapter.fetch_market_odds needs an odds CSV. Pass path=... or config={'odds_file': ...}. "
                "Note: most Kaggle NBA odds files are team markets, not player props."
            )
        odds = load_sportsbook_odds(path)
        return self.normalize_to_project_schema(odds, "market_odds")

    def fetch_closing_prices(self, event_id: str | None = None, market_types: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        from ..sportsbook_odds import load_sportsbook_odds, select_closing_odds

        path = kwargs.get("path", self.config.get("odds_file"))
        if not path:
            raise RuntimeError("KaggleCsvAdapter.fetch_closing_prices needs an odds CSV path with closing flags.")
        closing = select_closing_odds(load_sportsbook_odds(path))
        return self.normalize_to_project_schema(closing, "closing_prices")

    def normalize_to_project_schema(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return super().normalize_to_project_schema(pd.DataFrame(), entity)
        if entity in {"player_game_logs", "results"}:
            frame = apply_aliases(frame, BOX_SCORE_ALIASES)
            if "game_date" in frame.columns:
                frame = frame.copy()
                frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.normalize()
        elif entity in {"market_odds", "closing_prices"}:
            frame = self._normalize_team_odds(frame, entity)
        return super().normalize_to_project_schema(frame, entity)

    def _normalize_team_odds(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        output = frame.copy()
        output["market_type"] = "moneyline"
        output["stat_type"] = "winner"
        output["book"] = output.get("sportsbook_name")
        output["price"] = pd.to_numeric(output.get("home_moneyline"), errors="coerce")
        output["price_format"] = "american"
        output["implied_prob"] = pd.to_numeric(output.get("home_implied_prob"), errors="coerce")
        output["side"] = "home"
        output["game_date"] = pd.to_datetime(output.get("game_date"), errors="coerce").dt.normalize()
        if entity == "closing_prices":
            output["closing_timestamp"] = output.get("closing_timestamp")
            output["is_closing"] = True
        else:
            output["is_closing"] = output.get("is_closing", False)
        return output


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)
