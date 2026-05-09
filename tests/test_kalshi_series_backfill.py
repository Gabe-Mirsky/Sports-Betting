from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.kalshi_series_backfill import crawl_kalshi_series_markets  # noqa: E402


class FakeSeriesClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_historical_markets(self, params: dict) -> pd.DataFrame:
        self.calls.append(("historical", dict(params)))
        series = params["series_ticker"]
        if series == "KXNBAGAME":
            return pd.DataFrame(
                [
                    {
                        "market_ticker": "KXNBAGAME-25APR19MEMOKC-OKC",
                        "series_ticker": "KXNBAGAME",
                        "event_ticker": "KXNBAGAME-25APR19MEMOKC",
                        "market_title": "Memphis at Oklahoma City Winner?",
                        "yes_sub_title": "Oklahoma City",
                        "expected_expiration_time": "2025-04-19T20:00:00Z",
                    }
                ]
            )
        return pd.DataFrame()

    def get_markets(self, params: dict) -> pd.DataFrame:
        self.calls.append(("recent", dict(params)))
        return pd.DataFrame()


class TestKalshiSeriesBackfill(unittest.TestCase):
    def test_crawl_historical_series_writes_cache_and_possible_markets(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_kalshi_series_backfill"
        root.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        output_path = root / f"historical_series_markets_{suffix}.csv"
        possible_path = root / f"possible_{suffix}.csv"
        summary_path = root / f"summary_{suffix}.json"
        client = FakeSeriesClient()

        cached, possible, summary = crawl_kalshi_series_markets(
            series_tickers=["KXNBAGAME", "KXNBASPREAD"],
            client=client,
            max_pages=7,
            output_path=output_path,
            possible_output_path=possible_path,
            summary_path=summary_path,
            append=False,
            rebuild_possible_from_all_raw=False,
        )

        self.assertEqual(len(cached), 1)
        self.assertEqual(len(possible), 1)
        self.assertEqual(summary["cached_unique_markets"], 1)
        self.assertEqual(summary["possible_game_winner_rows"], 1)
        self.assertTrue(output_path.exists())
        self.assertTrue(possible_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertEqual(client.calls[0][1]["series_ticker"], "KXNBAGAME")
        self.assertEqual(client.calls[0][1]["max_pages"], 7)


if __name__ == "__main__":
    unittest.main()
