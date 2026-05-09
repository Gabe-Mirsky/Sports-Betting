"""Edge-threshold sweep utilities for paper-trading backtests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.backtest import run_backtest, summarize_backtest


def parse_thresholds(threshold_text: str) -> list[float]:
    """Parse comma-separated edge thresholds."""

    thresholds = []
    for item in threshold_text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0:
            raise ValueError("Edge thresholds must be non-negative.")
        thresholds.append(value)

    if not thresholds:
        raise ValueError("At least one threshold is required.")

    return sorted(set(thresholds))


def run_threshold_sweep(
    matched_markets_df: pd.DataFrame,
    thresholds: list[float],
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = 0.03,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
) -> pd.DataFrame:
    """Run the same matched markets through multiple edge thresholds."""

    rows: list[dict[str, Any]] = []
    for threshold in sorted(thresholds):
        trades = run_backtest(
            matched_markets_df,
            starting_bankroll=starting_bankroll,
            edge_threshold=threshold,
            max_bet_fraction=max_bet_fraction,
            min_market_price=min_market_price,
            max_market_price=max_market_price,
        )
        summary = summarize_backtest(trades, starting_bankroll=starting_bankroll)
        rows.append(
            {
                "edge_threshold": float(threshold),
                **summary,
            }
        )

    return pd.DataFrame(rows)
