from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.kalshi_candles import _extract_snapshot, download_candles_for_matches  # noqa: E402


class NoNetworkClient:
    def get_historical_cutoff(self) -> dict:
        raise AssertionError("future games should not request Kalshi candles")

    def get_market_candlesticks(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("future games should not request Kalshi candles")

    def get_historical_market_candlesticks(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("future games should not request Kalshi candles")


class TestKalshiCandles(unittest.TestCase):
    def test_snapshot_never_uses_candle_after_tipoff(self) -> None:
        game_start_ts = 1_700_000_000
        candles = pd.DataFrame(
            [
                {
                    "snapshot_ts": game_start_ts - 60 * 60,
                    "yes_bid": 54,
                    "yes_ask": 56,
                    "last_price": 55,
                    "volume": 100,
                    "open_interest": 20,
                    "period_interval": 1,
                },
                {
                    "snapshot_ts": game_start_ts + 60,
                    "yes_bid": 10,
                    "yes_ask": 12,
                    "last_price": 11,
                    "volume": 999,
                    "open_interest": 20,
                    "period_interval": 1,
                },
            ]
        )

        snapshot = _extract_snapshot(
            candles=candles,
            game_id="1",
            market_ticker="MKT",
            series_ticker="KXNBAGAME",
            game_start_ts=game_start_ts,
            snapshot_target="pregame_60m",
            minutes_before_tipoff=60,
            time_quality="exact",
        )

        self.assertEqual(snapshot["snapshot_ts"], game_start_ts - 60 * 60)
        self.assertEqual(snapshot["yes_price"], 56)
        self.assertEqual(snapshot["price_quality"], "bid_ask_available")

    def test_future_games_are_marked_missing_without_api_calls(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_kalshi_candles" / uuid.uuid4().hex
        matches = pd.DataFrame(
            [
                {
                    "game_id": "future-game",
                    "match_status": "auto_matched",
                    "market_ticker": "KXNBAGAME-99JAN01BOSNYK-BOS",
                    "series_ticker": "KXNBAGAME",
                }
            ]
        )
        games = pd.DataFrame(
            [
                {
                    "game_id": "future-game",
                    "game_date": "2099-01-01",
                }
            ]
        )

        prices = download_candles_for_matches(
            matches,
            games,
            client=NoNetworkClient(),
            candle_dir=root / "candles",
            output_path=root / "prices.csv",
        )

        self.assertEqual(len(prices), 3)
        self.assertEqual(set(prices["price_quality"]), {"missing"})
        self.assertTrue(prices["time_quality"].str.contains("future_game_not_downloaded").all())


if __name__ == "__main__":
    unittest.main()
