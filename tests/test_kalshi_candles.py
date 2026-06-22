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


class BatchOnlyClient:
    def __init__(self, snapshot_ts: int) -> None:
        self.snapshot_ts = snapshot_ts
        self.batch_calls = 0

    def get_historical_cutoff(self) -> dict:
        return {"cutoff_ts": 0}

    def get_batch_market_candlesticks(
        self,
        market_tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int,
        include_latest_before_start: bool = False,
    ) -> pd.DataFrame:
        self.batch_calls += 1
        if period_interval != 1:
            return pd.DataFrame()
        rows = []
        for index, ticker in enumerate(market_tickers):
            rows.append(
                {
                    "market_ticker": ticker,
                    "snapshot_ts": self.snapshot_ts,
                    "yes_bid": 50 + index,
                    "yes_ask": 52 + index,
                    "last_price": 51 + index,
                    "volume": 100,
                    "open_interest": 10,
                }
            )
        return pd.DataFrame(rows)

    def get_market_candlesticks(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("batch success should avoid per-ticker recent calls")

    def get_historical_market_candlesticks(self, *args, **kwargs) -> pd.DataFrame:
        raise AssertionError("batch success should avoid per-ticker historical calls")


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

        self.assertEqual(len(prices), 4)
        self.assertEqual(set(prices["price_quality"]), {"missing"})
        self.assertTrue(prices["time_quality"].str.contains("future_game_not_downloaded").all())

    def test_best_pregame_snapshot_uses_only_two_hour_window(self) -> None:
        game_start_ts = 1_700_000_000
        candles = pd.DataFrame(
            [
                {
                    "snapshot_ts": game_start_ts - 150 * 60,
                    "yes_bid": 40,
                    "yes_ask": 42,
                    "last_price": 41,
                    "period_interval": 1,
                },
                {
                    "snapshot_ts": game_start_ts - 80 * 60,
                    "yes_bid": 50,
                    "yes_ask": 52,
                    "last_price": 51,
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
            snapshot_target="pregame_best_le_120m",
            minutes_before_tipoff=0,
            time_quality="exact",
            max_minutes_before_tipoff=120,
        )

        self.assertEqual(snapshot["snapshot_ts"], game_start_ts - 80 * 60)
        self.assertEqual(snapshot["yes_price"], 52)
        self.assertEqual(snapshot["minutes_before_tipoff"], 80)

    def test_batch_candle_download_populates_per_ticker_prices(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_kalshi_candles" / uuid.uuid4().hex
        game_start = pd.Timestamp("2024-01-01 19:00:00", tz="America/New_York").tz_convert("UTC")
        snapshot_ts = int(game_start.timestamp()) - 60 * 60
        client = BatchOnlyClient(snapshot_ts=snapshot_ts)
        matches = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "match_status": "auto_matched",
                    "market_ticker": "MKT1",
                    "series_ticker": "KXNBAGAME",
                },
                {
                    "game_id": "g2",
                    "match_status": "auto_matched",
                    "market_ticker": "MKT2",
                    "series_ticker": "KXNBAGAME",
                },
            ]
        )
        games = pd.DataFrame(
            [
                {"game_id": "g1", "game_date": "2024-01-01", "game_start_time": game_start.isoformat()},
                {"game_id": "g2", "game_date": "2024-01-01", "game_start_time": game_start.isoformat()},
            ]
        )

        prices = download_candles_for_matches(
            matches,
            games,
            client=client,
            force=True,
            candle_dir=root / "candles",
            output_path=root / "prices.csv",
            use_batch=True,
        )

        self.assertGreaterEqual(client.batch_calls, 1)
        first_snapshots = prices[prices["snapshot_target"].eq("pregame_60m")].sort_values("market_ticker")
        self.assertEqual(first_snapshots["market_ticker"].tolist(), ["MKT1", "MKT2"])
        self.assertEqual(first_snapshots["price_quality"].tolist(), ["bid_ask_available", "bid_ask_available"])
        self.assertEqual(first_snapshots["yes_price"].tolist(), [52, 53])


if __name__ == "__main__":
    unittest.main()
