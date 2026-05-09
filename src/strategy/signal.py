"""YES-only trading signal generation."""

from __future__ import annotations

import pandas as pd

from strategy.market_math import cents_to_prob, expected_value_yes


def generate_yes_signal(
    model_yes_prob: float,
    yes_price_cents: float,
    edge_threshold: float = 0.05,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
) -> dict[str, object]:
    """Create a YES trade signal when model edge clears the threshold."""

    try:
        market_prob = cents_to_prob(yes_price_cents)
    except (TypeError, ValueError):
        return {
            "trade": False,
            "side": "",
            "model_prob": float(model_yes_prob),
            "market_prob": float("nan"),
            "edge": float("nan"),
            "price_cents": float("nan"),
            "expected_value": float("nan"),
            "reason": "invalid_market_price",
        }

    edge = float(model_yes_prob) - market_prob

    if market_prob < min_market_price:
        return {
            "trade": False,
            "side": "",
            "model_prob": float(model_yes_prob),
            "market_prob": market_prob,
            "edge": edge,
            "price_cents": float(yes_price_cents),
            "expected_value": expected_value_yes(model_yes_prob, yes_price_cents),
            "reason": "market_price_below_minimum",
        }

    if market_prob > max_market_price:
        return {
            "trade": False,
            "side": "",
            "model_prob": float(model_yes_prob),
            "market_prob": market_prob,
            "edge": edge,
            "price_cents": float(yes_price_cents),
            "expected_value": expected_value_yes(model_yes_prob, yes_price_cents),
            "reason": "market_price_above_maximum",
        }

    should_trade = edge >= edge_threshold
    return {
        "trade": bool(should_trade),
        "side": "YES" if should_trade else "",
        "model_prob": float(model_yes_prob),
        "market_prob": market_prob,
        "edge": edge,
        "price_cents": float(yes_price_cents),
        "expected_value": expected_value_yes(model_yes_prob, yes_price_cents),
        "reason": "edge_met" if should_trade else "edge_below_threshold",
    }


def add_yes_signals(
    markets: pd.DataFrame,
    edge_threshold: float = 0.05,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
) -> pd.DataFrame:
    """Add YES signal columns to matched market rows."""

    rows = [
        generate_yes_signal(
            row["model_yes_prob"],
            row["yes_mid_cents"],
            edge_threshold=edge_threshold,
            min_market_price=min_market_price,
            max_market_price=max_market_price,
        )
        for _, row in markets.iterrows()
    ]
    signal_df = pd.DataFrame(rows, index=markets.index)
    return pd.concat([markets.copy(), signal_df], axis=1)
