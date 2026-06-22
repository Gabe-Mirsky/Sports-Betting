"""Kalshi adapter: prediction-market prices, candlestick history, closing lines.

Reads the project's already-built Kalshi artifacts (pregame prices, matched
markets) and normalizes them to the canonical market/closing/result tables.
Kalshi contracts are Yes/No, so a single market maps to one priced ``side``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..source_adapter import SportsDataSourceAdapter


def _stat_type_from_ticker(ticker: str) -> str:
    """Map a market ticker prefix to a stat type (winner, points, rebounds...)."""

    from ..kalshi_taxonomy import PLAYER_PROP_TICKER_STAT_TYPES, WINNER_TICKER_PREFIXES

    text = str(ticker).upper()
    for prefix, stat_type in PLAYER_PROP_TICKER_STAT_TYPES.items():
        if text.startswith(prefix):
            return stat_type
    for prefix, stat_type in WINNER_TICKER_PREFIXES.items():
        if text.startswith(prefix):
            return stat_type
    return "unknown"


class KalshiAdapter(SportsDataSourceAdapter):
    """Prediction-market price source backed by cached Kalshi artifacts."""

    source_key = "kalshi"
    sport = "basketball"
    league = "NBA"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.processed_dir = Path(self.config.get("processed_dir", "data/processed"))
        self.reports_dir = Path(self.config.get("reports_dir", "data/reports"))
        self.pregame_prices_path = Path(
            self.config.get("pregame_prices_path", self.processed_dir / "kalshi_pregame_prices.csv")
        )
        self.matched_markets_path = Path(
            self.config.get("matched_markets_path", self.reports_dir / "matched_markets.csv")
        )

    def fetch_events(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, **kwargs: Any) -> pd.DataFrame:
        matched = _read_csv(self.matched_markets_path)
        if matched.empty:
            return self.normalize_to_project_schema(pd.DataFrame(), "events")
        events = matched.rename(columns={"market_ticker": "event_id"})
        return self.normalize_to_project_schema(events, "events")

    def fetch_market_odds(self, event_id: str | None = None, market_types: list[str] | None = None, snapshot_time: str | None = None, **kwargs: Any) -> pd.DataFrame:
        prices = _read_csv(self.pregame_prices_path)
        if prices.empty:
            return self.normalize_to_project_schema(pd.DataFrame(), "market_odds")
        if event_id is not None and "market_ticker" in prices.columns:
            prices = prices[prices["market_ticker"].astype(str).eq(str(event_id))]
        return self.normalize_to_project_schema(prices, "market_odds")

    def fetch_closing_prices(self, event_id: str | None = None, market_types: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        matched = _read_csv(self.matched_markets_path)
        if matched.empty or "clv_reference_price_cents" not in matched.columns:
            return self.normalize_to_project_schema(pd.DataFrame(), "closing_prices")
        if event_id is not None and "market_ticker" in matched.columns:
            matched = matched[matched["market_ticker"].astype(str).eq(str(event_id))]
        return self.normalize_to_project_schema(matched, "closing_prices")

    def fetch_results(self, season: int | None = None, start_date: str | None = None, end_date: str | None = None, **kwargs: Any) -> pd.DataFrame:
        matched = _read_csv(self.matched_markets_path)
        if matched.empty or "actual_yes_win" not in matched.columns:
            return self.normalize_to_project_schema(pd.DataFrame(), "results")
        return self.normalize_to_project_schema(matched, "results")

    def normalize_to_project_schema(self, frame: pd.DataFrame, entity: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return super().normalize_to_project_schema(pd.DataFrame(), entity)
        if entity == "market_odds":
            frame = self._normalize_market_odds(frame)
        elif entity == "closing_prices":
            frame = self._normalize_closing_prices(frame)
        elif entity == "results":
            frame = self._normalize_results(frame)
        elif entity == "events" and "event_id" not in frame.columns and "market_ticker" in frame.columns:
            frame = frame.rename(columns={"market_ticker": "event_id"})
        return super().normalize_to_project_schema(frame, entity)

    def _normalize_market_odds(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["market_id"] = output.get("market_ticker")
        output["event_id"] = output.get("market_ticker")
        output["market_type"] = "kalshi_contract"
        output["stat_type"] = output.get("market_ticker", pd.Series("", index=output.index)).map(_stat_type_from_ticker)
        output["side"] = "yes"
        output["price"] = pd.to_numeric(output.get("yes_ask"), errors="coerce")
        output["price_format"] = "cents_0_100"
        output["implied_prob"] = output["price"] / 100.0
        output["book"] = "kalshi"
        output["snapshot_time"] = pd.to_datetime(pd.to_numeric(output.get("snapshot_ts"), errors="coerce"), unit="s", errors="coerce")
        output["is_closing"] = False
        output["line"] = pd.NA
        return output

    def _normalize_closing_prices(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["market_id"] = output.get("market_ticker")
        output["event_id"] = output.get("market_ticker")
        output["market_type"] = "kalshi_contract"
        output["stat_type"] = output.get("market_ticker", pd.Series("", index=output.index)).map(_stat_type_from_ticker)
        output["side"] = "yes"
        output["price"] = pd.to_numeric(output.get("clv_reference_price_cents"), errors="coerce")
        output["price_format"] = "cents_0_100"
        output["implied_prob"] = output["price"] / 100.0
        output["book"] = "kalshi"
        output["closing_timestamp"] = output.get("clv_reference_snapshot")
        output["is_closing"] = True
        output["line"] = pd.NA
        return output

    def _normalize_results(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["event_id"] = output.get("market_ticker")
        output["stat_type"] = output.get("market_ticker", pd.Series("", index=output.index)).map(_stat_type_from_ticker)
        actual_yes = output.get("actual_yes_win")
        output["actual_value"] = pd.to_numeric(actual_yes, errors="coerce")
        output["outcome"] = actual_yes.map(lambda value: "yes" if bool(value) else "no") if actual_yes is not None else pd.NA
        output["settlement_price"] = pd.NA
        return output


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
