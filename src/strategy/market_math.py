"""Math helpers for Kalshi-style binary contracts."""

from __future__ import annotations

import math


def cents_to_prob(price_cents: float) -> float:
    """Convert contract cents to implied probability."""

    price = float(price_cents)
    if math.isnan(price):
        raise ValueError("price_cents cannot be NaN")
    if price < 0 or price > 100:
        raise ValueError(f"price_cents must be between 0 and 100: {price_cents}")
    return price / 100.0


def prob_to_fair_cents(prob: float) -> float:
    """Convert fair probability to fair cents."""

    probability = float(prob)
    if math.isnan(probability):
        raise ValueError("prob cannot be NaN")
    if probability < 0 or probability > 1:
        raise ValueError(f"prob must be between 0 and 1: {prob}")
    return probability * 100.0


def expected_value_yes(model_prob: float, yes_price_cents: float) -> float:
    """Expected dollars per YES share before fees."""

    cost = cents_to_prob(yes_price_cents)
    return float(model_prob) - cost


def expected_value_no(model_prob_yes: float, no_price_cents: float) -> float:
    """Expected dollars per NO share before fees."""

    no_cost = cents_to_prob(no_price_cents)
    return (1.0 - float(model_prob_yes)) - no_cost


def payout_profit_if_yes_wins(yes_price_cents: float, shares: int) -> float:
    """Profit when bought YES shares resolve YES."""

    cost = cents_to_prob(yes_price_cents)
    return int(shares) * (1.0 - cost)


def payout_profit_if_yes_loses(yes_price_cents: float, shares: int) -> float:
    """Profit when bought YES shares resolve NO."""

    cost = cents_to_prob(yes_price_cents)
    return -int(shares) * cost
