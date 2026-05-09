from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.market_quality import analyze_market_data_quality  # noqa: E402


class TestMarketQuality(unittest.TestCase):
    def test_quality_report_warns_on_close_price_and_small_sample(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT1",
                    "yes_mid_cents": 55,
                    "price_source": "close_price",
                    "settlement": "YES",
                    "volume": 100,
                }
            ]
        )
        report = analyze_market_data_quality(markets, min_sample_size=30)
        self.assertEqual(report["price_quality"]["close_price_only_count"], 1)
        self.assertTrue(any("Small sample size" in warning for warning in report["warnings"]))
        self.assertTrue(any("close_price_cents" in warning for warning in report["warnings"]))

    def test_quality_report_warns_on_missing_spread_and_unresolved_settlement(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT2",
                    "yes_mid_cents": 50,
                    "price_source": "yes_mid",
                    "settlement": "",
                    "volume": None,
                }
            ]
        )
        report = analyze_market_data_quality(markets, min_sample_size=1)
        self.assertEqual(report["spread_quality"]["rows_missing_bid_or_ask"], 1)
        self.assertEqual(report["settlement_quality"]["unresolved_count"], 1)
        self.assertEqual(report["volume_quality"]["missing_volume_count"], 1)
        self.assertTrue(any("missing bid/ask" in warning for warning in report["warnings"]))
        self.assertTrue(any("unresolved settlement" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
