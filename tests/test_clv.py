from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.clv import build_clv_reports  # noqa: E402


class TestCLV(unittest.TestCase):
    def test_build_clv_reports_groups_traded_rows(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "trade": True,
                    "date": "2025-01-01",
                    "yes_team_abbr": "BOS",
                    "side": "YES",
                    "edge": 0.06,
                    "price_cents": 50,
                    "volume": 200,
                    "clv_cents": 4,
                },
                {
                    "trade": True,
                    "date": "2025-01-02",
                    "yes_team_abbr": "NYK",
                    "side": "NO",
                    "edge": 0.09,
                    "price_cents": 70,
                    "volume": 5,
                    "clv_cents": -2,
                },
                {
                    "trade": False,
                    "date": "2025-01-03",
                    "yes_team_abbr": "BOS",
                    "edge": 0.01,
                    "price_cents": 40,
                    "volume": 100,
                    "clv_cents": 10,
                },
            ]
        )

        reports, summary = build_clv_reports(trades)

        self.assertEqual(summary["trades_with_clv"], 2)
        self.assertAlmostEqual(summary["avg_clv_cents"], 1.0)
        self.assertAlmostEqual(summary["median_clv_cents"], 1.0)
        self.assertAlmostEqual(summary["positive_clv_rate"], 0.5)
        self.assertIn("by_edge_bucket", reports)
        self.assertIn("by_price_bucket", reports)
        self.assertIn("by_team", reports)
        self.assertIn("by_side", reports)
        self.assertIn("by_liquidity", reports)
        self.assertEqual(set(reports["by_side"]["side"]), {"YES", "NO"})


if __name__ == "__main__":
    unittest.main()
