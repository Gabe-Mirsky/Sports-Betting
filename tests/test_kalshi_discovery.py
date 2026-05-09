from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.kalshi_discovery import filter_broad_nba_markets  # noqa: E402


class TestKalshiDiscovery(unittest.TestCase):
    def test_broad_filter_keeps_winners_spreads_totals_and_props(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "WINNER-1",
                    "market_title": "Will the Boston Celtics beat the New York Knicks?",
                },
                {
                    "market_ticker": "SPREAD-1",
                    "market_title": "Will Boston cover the -4.5 spread against New York?",
                },
                {
                    "market_ticker": "TOTAL-1",
                    "market_title": "Will Boston vs New York total points go over 221.5?",
                },
                {
                    "market_ticker": "PROP-1",
                    "market_title": "Will Jayson Tatum score over 27.5 points?",
                },
                {
                    "market_ticker": "OTHER-1",
                    "market_title": "Will the federal funds rate change this month?",
                },
                {
                    "market_ticker": "CRYPTO-1",
                    "market_title": "Will BNB price increase on Coinbase?",
                    "rules_secondary": "Checking Coinbase may help guide your decision.",
                },
            ]
        )

        filtered = filter_broad_nba_markets(markets, player_names=["Jayson Tatum"])
        tickers = set(filtered["market_ticker"])

        self.assertEqual(tickers, {"WINNER-1", "SPREAD-1", "TOTAL-1", "PROP-1"})
        reasons = filtered.set_index("market_ticker")["nba_discovery_reason"].to_dict()
        self.assertIn("team", reasons["WINNER-1"])
        self.assertIn("team", reasons["SPREAD-1"])
        self.assertIn("team", reasons["TOTAL-1"])
        self.assertIn("player_name", reasons["PROP-1"])

    def test_generic_finals_without_nba_signal_is_not_enough(self) -> None:
        markets = pd.DataFrame(
            [
                {"market_ticker": "FINAL-1", "market_title": "Will Game 7 of the finals go to overtime?"},
                {"market_ticker": "NBA-1", "market_title": "Will the NBA finals have a Game 7?"},
            ]
        )

        filtered = filter_broad_nba_markets(markets)

        self.assertEqual(set(filtered["market_ticker"]), {"NBA-1"})


if __name__ == "__main__":
    unittest.main()
