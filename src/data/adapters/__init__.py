"""Concrete sport data-source adapters and a small registry.

Adapters implement the generic ``SportsDataSourceAdapter`` contract. They wrap the
project's existing clients where data already flows (nba_api, Kalshi, Kaggle CSV)
and declare-only where integration is still planned (Odds API, Basketball-Ref).
"""

from __future__ import annotations

from ..source_adapter import SportsDataSourceAdapter
from .basketball_reference_adapter import BasketballReferenceAdapter
from .kaggle_csv_adapter import KaggleCsvAdapter
from .kalshi_adapter import KalshiAdapter
from .nba_api_adapter import NbaApiAdapter
from .odds_api_adapter import OddsApiAdapter


ADAPTER_REGISTRY: dict[str, type[SportsDataSourceAdapter]] = {
    "nba_api": NbaApiAdapter,
    "kalshi": KalshiAdapter,
    "odds_api": OddsApiAdapter,
    "kaggle_csv": KaggleCsvAdapter,
    "basketball_reference": BasketballReferenceAdapter,
}


def get_adapter(key: str, config: dict | None = None) -> SportsDataSourceAdapter:
    """Instantiate the adapter registered under ``key``."""

    if key not in ADAPTER_REGISTRY:
        raise KeyError(f"No adapter registered for '{key}'. Known: {sorted(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[key](config=config)


__all__ = [
    "ADAPTER_REGISTRY",
    "get_adapter",
    "NbaApiAdapter",
    "KalshiAdapter",
    "OddsApiAdapter",
    "KaggleCsvAdapter",
    "BasketballReferenceAdapter",
]
