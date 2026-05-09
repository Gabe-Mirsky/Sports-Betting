from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.market_math import (  # noqa: E402
    cents_to_prob,
    expected_value_yes,
    payout_profit_if_yes_loses,
    payout_profit_if_yes_wins,
    prob_to_fair_cents,
)


class TestMarketMath(unittest.TestCase):
    def test_cents_maps_to_probability(self) -> None:
        self.assertAlmostEqual(cents_to_prob(52), 0.52)

    def test_probability_maps_to_fair_cents(self) -> None:
        self.assertAlmostEqual(prob_to_fair_cents(0.61), 61.0)

    def test_expected_value_is_model_probability_minus_price(self) -> None:
        self.assertAlmostEqual(expected_value_yes(0.60, 50), 0.10)

    def test_negative_expected_value(self) -> None:
        self.assertLess(expected_value_yes(0.45, 55), 0)

    def test_yes_profit_helpers(self) -> None:
        self.assertAlmostEqual(payout_profit_if_yes_wins(50, 2), 1.0)
        self.assertAlmostEqual(payout_profit_if_yes_loses(50, 2), -1.0)


if __name__ == "__main__":
    unittest.main()
