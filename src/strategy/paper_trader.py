"""Paper-trading helpers that never place real orders."""

from __future__ import annotations

import pandas as pd

from strategy.signal import add_yes_signals


def suggest_paper_trades(
    matched_markets: pd.DataFrame,
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    """Return paper-trade suggestions from matched market rows."""

    suggestions = add_yes_signals(matched_markets, edge_threshold=edge_threshold)
    return suggestions[suggestions["trade"]].copy()
