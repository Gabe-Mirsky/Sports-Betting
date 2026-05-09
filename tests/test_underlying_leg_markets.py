from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.underlying_leg_markets import fetch_underlying_leg_markets  # noqa: E402


class FakeClient:
    def get_market(self, ticker: str) -> pd.DataFrame:
        if ticker.startswith("KXNBASPREAD"):
            return pd.DataFrame([{"market_ticker": ticker, "market_title": "Spread"}])
        if ticker.startswith("KXNBATOTAL"):
            return pd.DataFrame([{"market_ticker": ticker, "market_title": "Total"}])
        return pd.DataFrame()

    def get_historical_market(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame()


class TestUnderlyingLegMarkets(unittest.TestCase):
    def test_fetch_underlying_leg_markets_skips_game_winners_by_default(self) -> None:
        legs = pd.DataFrame(
            [
                {"leg_market_ticker": "KXNBAGAME-26MAY08NYKBOS-BOS", "leg_category": "game_winner"},
                {"leg_market_ticker": "KXNBASPREAD-26MAY08NYKBOS-BOS5", "leg_category": "spread_handicap"},
            ]
        )

        fetched, requests, summary = fetch_underlying_leg_markets(legs, client=FakeClient())

        self.assertEqual(len(requests), 1)
        self.assertEqual(fetched.loc[0, "market_ticker"], "KXNBASPREAD-26MAY08NYKBOS-BOS5")
        self.assertEqual(summary["fetched_rows"], 1)
        self.assertFalse(summary["include_game_winners"])

    def test_fetch_underlying_leg_markets_prioritizes_spreads_and_totals(self) -> None:
        legs = pd.DataFrame(
            [
                {"leg_market_ticker": "KXNBAPLAYER-26MAY08BOS-TATUM25", "leg_category": "player_points_rebounds_assists"},
                {"leg_market_ticker": "KXNBATOTAL-26MAY08NYKBOS-O215", "leg_category": "total_points_over_under"},
                {"leg_market_ticker": "KXNBASPREAD-26MAY08NYKBOS-BOS5", "leg_category": "spread_handicap"},
            ]
        )

        _, requests, summary = fetch_underlying_leg_markets(legs, client=FakeClient())

        self.assertEqual(
            requests["market_ticker"].tolist(),
            [
                "KXNBASPREAD-26MAY08NYKBOS-BOS5",
                "KXNBATOTAL-26MAY08NYKBOS-O215",
                "KXNBAPLAYER-26MAY08BOS-TATUM25",
            ],
        )
        self.assertEqual(summary["fetched_rows"], 2)

    def test_fetch_underlying_leg_markets_stops_after_consecutive_failures(self) -> None:
        legs = pd.DataFrame(
            [
                {"leg_market_ticker": "KXNBASPREAD-26MAY08NYKBOS-BOS5", "leg_category": "spread_handicap"},
                {"leg_market_ticker": "KXNBATOTAL-26MAY08NYKBOS-O215", "leg_category": "total_points_over_under"},
                {"leg_market_ticker": "KXNBAPLAYER-26MAY08BOS-TATUM25", "leg_category": "player_points_rebounds_assists"},
                {"leg_market_ticker": "KXNBAPLAYER-26MAY08BOS-BROWN20", "leg_category": "player_points_rebounds_assists"},
            ]
        )

        _, requests, summary = fetch_underlying_leg_markets(
            legs,
            client=FakeClient(),
            max_consecutive_failures=1,
        )

        self.assertEqual(len(requests), 3)
        self.assertTrue(summary["stopped_after_consecutive_failures"])
        self.assertEqual(summary["attempted_tickers"], 3)


if __name__ == "__main__":
    unittest.main()
