"""YES-only trading signal generation."""

from __future__ import annotations

import pandas as pd

from strategy.market_math import cents_to_prob, expected_value_no, expected_value_yes


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


def _no_ask_from_yes_bid(yes_bid_cents: float | None) -> float | None:
    if yes_bid_cents is None:
        return None
    try:
        yes_bid = float(yes_bid_cents)
    except (TypeError, ValueError):
        return None
    if pd.isna(yes_bid):
        return None
    return 100.0 - yes_bid


def generate_two_sided_signal(
    model_yes_prob: float,
    yes_ask_cents: float,
    yes_bid_cents: float | None = None,
    edge_threshold: float = 0.05,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
    allow_no: bool = True,
) -> dict[str, object]:
    """Create the best tradable YES or NO signal from bid/ask prices."""

    candidates: list[dict[str, object]] = []
    try:
        yes_market_prob = cents_to_prob(yes_ask_cents)
        candidates.append(
            {
                "side": "YES",
                "model_prob": float(model_yes_prob),
                "market_prob": yes_market_prob,
                "edge": float(model_yes_prob) - yes_market_prob,
                "price_cents": float(yes_ask_cents),
                "expected_value": expected_value_yes(model_yes_prob, yes_ask_cents),
            }
        )
    except (TypeError, ValueError):
        pass

    no_ask_cents = _no_ask_from_yes_bid(yes_bid_cents)
    if allow_no and no_ask_cents is not None:
        try:
            no_market_prob = cents_to_prob(no_ask_cents)
            candidates.append(
                {
                    "side": "NO",
                    "model_prob": 1.0 - float(model_yes_prob),
                    "market_prob": no_market_prob,
                    "edge": (1.0 - float(model_yes_prob)) - no_market_prob,
                    "price_cents": float(no_ask_cents),
                    "expected_value": expected_value_no(model_yes_prob, no_ask_cents),
                }
            )
        except (TypeError, ValueError):
            pass

    if not candidates:
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

    best = max(candidates, key=lambda item: float(item["edge"]))
    market_prob = float(best["market_prob"])
    edge = float(best["edge"])
    candidate_side = str(best["side"])
    if market_prob < min_market_price:
        reason = "market_price_below_minimum"
        trade = False
        side = ""
    elif market_prob > max_market_price:
        reason = "market_price_above_maximum"
        trade = False
        side = ""
    elif edge >= edge_threshold:
        reason = "edge_met"
        trade = True
        side = str(best["side"])
    else:
        reason = "edge_below_threshold"
        trade = False
        side = ""

    return {
        "trade": bool(trade),
        "side": side,
        "candidate_side": candidate_side,
        "model_prob": float(best["model_prob"]),
        "market_prob": market_prob,
        "edge": edge,
        "price_cents": float(best["price_cents"]),
        "expected_value": float(best["expected_value"]),
        "reason": reason,
    }


def add_two_sided_signals(
    markets: pd.DataFrame,
    edge_threshold: float = 0.05,
    min_market_price: float = 0.05,
    max_market_price: float = 0.95,
    allow_no: bool = True,
) -> pd.DataFrame:
    """Add best-side YES/NO signal columns to matched market rows."""

    rows = [
        generate_two_sided_signal(
            row["model_yes_prob"],
            row.get("yes_ask", row.get("yes_mid_cents")),
            yes_bid_cents=row.get("yes_bid"),
            edge_threshold=edge_threshold,
            min_market_price=min_market_price,
            max_market_price=max_market_price,
            allow_no=allow_no,
        )
        for _, row in markets.iterrows()
    ]
    signal_df = pd.DataFrame(rows, index=markets.index)
    return pd.concat([markets.copy(), signal_df], axis=1)
