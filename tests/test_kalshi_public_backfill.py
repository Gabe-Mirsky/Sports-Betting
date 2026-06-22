from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.kalshi_public_backfill import _optional_status_params, backfill_public_sports_nba_markets  # noqa: E402
from data.kalshi_client import _batch_candlestick_payload_to_frame  # noqa: E402


class FakeKalshiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_params = dict(params or {})
        self.calls.append((path, request_params))
        if path == "/series":
            return {
                "series": [
                    {
                        "ticker": "KXNBAGAME",
                        "title": "NBA game winner",
                        "category": "Sports",
                    },
                    {
                        "ticker": "KXWEATHER",
                        "title": "Weather",
                        "category": "Climate",
                    },
                ]
            }
        if path == "/events":
            return {
                "events": [
                    {
                        "event_ticker": "KXNBAGAME-26MAR01BOSNYK",
                        "series_ticker": "KXNBAGAME",
                        "title": "Boston Celtics at New York Knicks",
                    }
                ]
            }
        if path == "/markets":
            return {
                "markets": [
                    {
                        "ticker": "KXNBAGAME-26MAR01BOSNYK-BOS",
                        "event_ticker": "KXNBAGAME-26MAR01BOSNYK",
                        "series_ticker": "KXNBAGAME",
                        "title": "Will the Boston Celtics beat the New York Knicks?",
                        "status": "settled",
                        "close_time": "2026-03-02T00:30:00Z",
                    }
                ]
            }
        if path == "/historical/markets":
            return {
                "markets": [
                    {
                        "ticker": "KXNBAGAME-26FEB01LALBOS-BOS",
                        "event_ticker": "KXNBAGAME-26FEB01LALBOS",
                        "series_ticker": "KXNBAGAME",
                        "title": "Los Angeles Lakers vs Boston Winner?",
                        "status": "finalized",
                        "close_time": "2026-02-02T00:30:00Z",
                    }
                ]
            }
        return {}


class TestKalshiPublicBackfill(unittest.TestCase):
    def test_backfill_public_sports_caches_raw_and_writes_possible_markets(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_public_backfill"
        if root.exists():
            shutil.rmtree(root)
        try:
            result = backfill_public_sports_nba_markets(
                client=FakeKalshiClient(),  # type: ignore[arg-type]
                raw_dir=root / "raw",
                output_dir=root / "tables",
                force=True,
                max_pages=5,
            )
            summary = result["summary"]

            self.assertEqual(summary["sports_series"], 2)
            self.assertEqual(summary["nba_series"], 1)
            self.assertEqual(summary["nba_events"], 1)
            self.assertEqual(summary["nba_markets"], 2)
            self.assertEqual(summary["possible_nba_markets"], 2)
            self.assertTrue((root / "raw" / "json").exists())
            self.assertGreater(len(list((root / "raw" / "json").glob("*.json"))), 0)
            self.assertTrue((root / "tables" / "nba_markets.csv").exists())
            self.assertEqual(
                result["possible_nba_markets"]["market_ticker"].iloc[0],
                "KXNBAGAME-26MAR01BOSNYK-BOS",
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_backfill_can_limit_series_and_events(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_public_backfill_limited"
        if root.exists():
            shutil.rmtree(root)
        try:
            result = backfill_public_sports_nba_markets(
                client=FakeKalshiClient(),  # type: ignore[arg-type]
                raw_dir=root / "raw",
                output_dir=root / "tables",
                force=True,
                max_pages=5,
                series_tickers=["KXNBAGAME"],
                max_events_per_series=1,
                sleep_seconds=0.0,
            )

            self.assertEqual(result["summary"]["nba_series"], 1)
            self.assertEqual(result["summary"]["nba_events"], 1)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_batch_candlestick_payload_keeps_market_ticker(self) -> None:
        frame = _batch_candlestick_payload_to_frame(
            {
                "markets": [
                    {
                        "market_ticker": "KXNBAGAME-26MAR01BOSNYK-BOS",
                        "candlesticks": [
                            {
                                "end_period_ts": 1772411400,
                                "yes_bid": {"close_dollars": "0.41"},
                                "yes_ask": {"close_dollars": "0.43"},
                                "price": {"close_dollars": "0.42"},
                                "volume_fp": "10",
                                "open_interest_fp": "20",
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["market_ticker"].iloc[0], "KXNBAGAME-26MAR01BOSNYK-BOS")
        self.assertEqual(float(frame["yes_bid"].iloc[0]), 0.41)
        self.assertEqual(float(frame["yes_ask"].iloc[0]), 0.43)

    def test_optional_status_params_can_omit_status_filter(self) -> None:
        self.assertEqual(_optional_status_params("settled"), {"status": "settled"})
        self.assertEqual(_optional_status_params("all"), {})
        self.assertEqual(_optional_status_params(""), {})


if __name__ == "__main__":
    unittest.main()
