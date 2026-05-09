"""Bankroll sizing helpers for paper trading."""

from __future__ import annotations

import math

from strategy.market_math import cents_to_prob


def calculate_flat_fractional_shares(
    bankroll: float,
    price_cents: float,
    max_bet_fraction: float = 0.03,
) -> int:
    """Return whole shares affordable under a flat bankroll fraction cap."""

    bankroll = float(bankroll)
    if bankroll <= 0:
        return 0

    contract_cost = cents_to_prob(price_cents)
    if contract_cost <= 0:
        return 0

    max_spend = min(bankroll, bankroll * float(max_bet_fraction))
    shares = math.floor(max_spend / contract_cost)
    total_cost = shares * contract_cost
    if total_cost > bankroll:
        shares = math.floor(bankroll / contract_cost)
    return max(0, int(shares))


def fractional_kelly_fraction(
    model_prob: float,
    price_cents: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Return capped fractional Kelly bankroll fraction for a YES share."""

    cost = cents_to_prob(price_cents)
    if cost <= 0 or cost >= 1:
        return 0.0

    p = float(model_prob)
    q = 1.0 - p
    b = (1.0 - cost) / cost
    full_kelly = (b * p - q) / b
    return max(0.0, full_kelly * float(kelly_fraction))
