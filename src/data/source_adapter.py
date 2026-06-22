"""Sport-agnostic data-source adapter interface and canonical project schemas.

This module defines the contract every data source must implement so that future
sports (or new vendors) can plug into the same pipeline. It does not fetch the
full model's data; it standardizes how sources are *described* and how their raw
rows are *normalized* into project-canonical tables.

The seven adapter methods mirror the stages the prediction pipeline needs:

    fetch_events            -> games/contests to model
    fetch_players           -> player identities for prop modeling
    fetch_player_game_logs  -> historical per-player box-score actuals
    fetch_market_odds       -> market/sportsbook prices (incl. player props)
    fetch_closing_prices    -> the last tradable price before tip-off (for CLV)
    fetch_results           -> settled outcomes used to grade picks
    normalize_to_project_schema -> map raw rows onto a canonical table

Concrete adapters live in ``src.data.adapters`` and only override the methods
their source can actually support; everything else raises a clear, typed error.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import pandas as pd


# Canonical entity tables every sport normalizes into. Column order is stable so
# downstream feature code and tests can rely on it.
ENTITY_SCHEMAS: dict[str, list[str]] = {
    "events": [
        "source",
        "sport",
        "league",
        "event_id",
        "game_id",
        "season",
        "game_date",
        "start_time",
        "home_team_abbr",
        "away_team_abbr",
        "game_key",
        "status",
    ],
    "players": [
        "source",
        "sport",
        "league",
        "player_id",
        "player_name",
        "team_abbr",
        "position",
        "season",
    ],
    "player_game_logs": [
        "source",
        "sport",
        "league",
        "player_id",
        "player_name",
        "game_id",
        "game_date",
        "season",
        "team_abbr",
        "opponent_abbr",
        "is_home",
        "minutes",
        "points",
        "rebounds",
        "assists",
        "threes",
        "blocks",
        "steals",
        "turnovers",
    ],
    "market_odds": [
        "source",
        "sport",
        "league",
        "event_id",
        "game_id",
        "game_date",
        "market_id",
        "market_type",
        "player_id",
        "player_name",
        "stat_type",
        "line",
        "side",
        "price",
        "price_format",
        "implied_prob",
        "book",
        "snapshot_time",
        "is_closing",
    ],
    "closing_prices": [
        "source",
        "sport",
        "league",
        "event_id",
        "game_id",
        "game_date",
        "market_id",
        "market_type",
        "player_id",
        "player_name",
        "stat_type",
        "line",
        "side",
        "price",
        "price_format",
        "implied_prob",
        "book",
        "closing_timestamp",
        "is_closing",
    ],
    "results": [
        "source",
        "sport",
        "league",
        "event_id",
        "game_id",
        "game_date",
        "season",
        "player_id",
        "player_name",
        "stat_type",
        "actual_value",
        "outcome",
        "settlement_price",
    ],
}

# The seven capability slots an adapter can advertise.
ADAPTER_METHODS = (
    "fetch_events",
    "fetch_players",
    "fetch_player_game_logs",
    "fetch_market_odds",
    "fetch_closing_prices",
    "fetch_results",
    "normalize_to_project_schema",
)


class UnsupportedCapabilityError(NotImplementedError):
    """Raised when an adapter is asked for data its source cannot provide."""


def coerce_to_project_schema(
    frame: pd.DataFrame,
    entity: str,
    source: str,
    sport: str = "basketball",
    league: str = "NBA",
) -> pd.DataFrame:
    """Reindex an already-renamed frame onto the canonical columns for ``entity``.

    Adapters first rename their raw columns to canonical names, then call this to
    guarantee a stable, fully-populated column set. Unknown source columns are
    dropped; missing canonical columns are added as NA.
    """

    if entity not in ENTITY_SCHEMAS:
        raise KeyError(f"Unknown entity '{entity}'. Expected one of {sorted(ENTITY_SCHEMAS)}.")
    columns = ENTITY_SCHEMAS[entity]
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        output[column] = frame[column] if column in frame.columns else pd.NA
    output["source"] = source
    output["sport"] = sport
    output["league"] = league
    return output.reset_index(drop=True)


class SportsDataSourceAdapter(ABC):
    """Generic, sport-agnostic interface for a single data source.

    Subclasses set ``source_key`` (matching an entry in ``source_catalog``) plus
    ``sport``/``league`` and override only the fetch methods their source supports.
    Unsupported methods inherit the default body that raises
    :class:`UnsupportedCapabilityError`, so capability gaps fail loudly.
    """

    source_key: str = ""
    sport: str = "basketball"
    league: str = "NBA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    # --- capability description -------------------------------------------------
    @property
    def capability(self) -> Any:
        """Return the catalog record describing this source (priority, limits...)."""

        from .source_catalog import get_source

        return get_source(self.source_key)

    def supports(self, method: str) -> bool:
        """Whether this adapter advertises ``method`` in the source catalog."""

        try:
            return method in self.capability.adapter_capabilities
        except KeyError:
            return False

    def _unsupported(self, method: str) -> "UnsupportedCapabilityError":
        return UnsupportedCapabilityError(
            f"{type(self).__name__} ({self.source_key}) does not support {method}(). "
            "See data/reports/data_source_audit.md for the capability matrix."
        )

    # --- fetch contract (override where supported) ------------------------------
    def fetch_events(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return games/contests to model (schedule + teams)."""

        raise self._unsupported("fetch_events")

    def fetch_players(self, season: int | None = None, team: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return player identities for the league/season."""

        raise self._unsupported("fetch_players")

    def fetch_player_game_logs(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, player_id: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return per-player, per-game box-score actuals."""

        raise self._unsupported("fetch_player_game_logs")

    def fetch_market_odds(self, event_id: str | None = None, market_types: list[str] | None = None, snapshot_time: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return market/sportsbook prices, including player props where available."""

        raise self._unsupported("fetch_market_odds")

    def fetch_closing_prices(self, event_id: str | None = None, market_types: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return the last tradable price before tip-off (for closing-line value)."""

        raise self._unsupported("fetch_closing_prices")

    def fetch_results(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return settled outcomes used to grade picks."""

        raise self._unsupported("fetch_results")

    # --- normalization ----------------------------------------------------------
    def normalize_to_project_schema(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        """Map a raw/renamed frame onto the canonical table for ``entity``.

        The default implementation assumes columns already carry canonical names
        and simply coerces them to the schema. Adapters with vendor-specific column
        names should rename first, then call ``super().normalize_to_project_schema``.
        """

        if frame is None or frame.empty:
            return coerce_to_project_schema(pd.DataFrame(), entity, self.source_key, self.sport, self.league)
        return coerce_to_project_schema(frame, entity, self.source_key, self.sport, self.league)
