"""The Odds API adapter: current multi-book sportsbook lines and player props.

Integration is planned, not wired into the pipeline yet. The flatten + normalize
path is fully implemented and testable; the live HTTP fetch is gated behind an
``ODDS_API_KEY`` so the rest of the project works offline. Free tier is ~500
requests/month and player-prop markets cost extra credits, so fetches are meant
to be sparse and snapshot-driven (poll near tip-off to approximate closing lines).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from ..source_adapter import SportsDataSourceAdapter


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "basketball_nba"
# Featured player-prop market keys (each costs extra credits on the free tier).
DEFAULT_PLAYER_PROP_MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
)


def flatten_odds_api_events(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten The Odds API event-odds JSON into one row per outcome.

    Accepts the documented ``/sports/{sport}/odds`` and event-odds shape where each
    event has bookmakers -> markets -> outcomes. Player-prop outcomes carry a
    ``description`` (player name) and ``point`` (line).
    """

    rows: list[dict[str, Any]] = []
    for event in events or []:
        event_id = event.get("id")
        commence = event.get("commence_time")
        home = event.get("home_team")
        away = event.get("away_team")
        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key")
            updated = bookmaker.get("last_update")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event_id,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "book": book,
                            "snapshot_time": updated,
                            "market_key": market_key,
                            "outcome_name": outcome.get("name"),
                            "player_name": outcome.get("description"),
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                        }
                    )
    return pd.DataFrame(rows)


def _decimal_to_implied_prob(value: Any) -> float:
    odds = pd.to_numeric(value, errors="coerce")
    if pd.isna(odds) or odds <= 1.0:
        return float("nan")
    return 1.0 / float(odds)


class OddsApiAdapter(SportsDataSourceAdapter):
    """Sportsbook player-prop and game-line source (planned live integration)."""

    source_key = "odds_api"
    sport = "basketball"
    league = "NBA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_key = self.config.get("api_key") or os.getenv("ODDS_API_KEY")
        self.sport_key = self.config.get("sport_key", DEFAULT_SPORT_KEY)
        self.regions = self.config.get("regions", "us")

    def fetch_market_odds(
        self,
        event_id: str | None = None,
        market_types: list[str] | None = None,
        snapshot_time: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch current odds for the sport. Requires ODDS_API_KEY (quota-limited)."""

        if not self.api_key:
            raise RuntimeError(
                "OddsApiAdapter.fetch_market_odds requires an ODDS_API_KEY. Set the env var or pass "
                "config={'api_key': ...}. Free tier is ~500 req/month; prop markets cost extra credits."
            )
        markets = ",".join(market_types or DEFAULT_PLAYER_PROP_MARKETS)
        params = {"apiKey": self.api_key, "regions": self.regions, "markets": markets, "oddsFormat": "decimal"}
        path = f"/sports/{self.sport_key}/odds"
        if event_id:
            path = f"/sports/{self.sport_key}/events/{event_id}/odds"
        payload = self._get_json(path, params)
        events = payload if isinstance(payload, list) else [payload]
        return self.normalize_to_project_schema(flatten_odds_api_events(events), "market_odds")

    def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        url = f"{ODDS_API_BASE}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "nba-kalshi-predictor/research"})
        with urllib.request.urlopen(request, timeout=self.config.get("timeout", 20)) as response:
            return json.loads(response.read().decode("utf-8"))

    def normalize_to_project_schema(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return super().normalize_to_project_schema(pd.DataFrame(), entity)
        if entity == "market_odds":
            frame = self._normalize_market_odds(frame)
        return super().normalize_to_project_schema(frame, entity)

    def _normalize_market_odds(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        market_key = output.get("market_key", pd.Series("", index=output.index)).astype(str)
        output["market_id"] = market_key
        output["market_type"] = market_key
        output["stat_type"] = market_key.str.replace("player_", "", regex=False)
        output["line"] = pd.to_numeric(output.get("point"), errors="coerce")
        output["side"] = output.get("outcome_name", pd.Series("", index=output.index)).astype(str).str.lower()
        output["price"] = pd.to_numeric(output.get("price"), errors="coerce")
        output["price_format"] = "decimal"
        output["implied_prob"] = output["price"].map(_decimal_to_implied_prob)
        output["game_date"] = pd.to_datetime(output.get("commence_time"), errors="coerce").dt.normalize()
        output["snapshot_time"] = pd.to_datetime(output.get("snapshot_time"), errors="coerce")
        output["is_closing"] = False
        return output
