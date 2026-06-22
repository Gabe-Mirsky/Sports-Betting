from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.edge_failure import build_edge_failure_diagnosis  # noqa: E402


class TestEdgeFailureDiagnosis(unittest.TestCase):
    def test_edge_failure_ranks_bad_segments(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "market_ticker": "A",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 20,
                    "edge": 0.08,
                    "calibrated_expected_roi": 1.5,
                    "volume": 100,
                    "clv_cents": -3,
                    "realized_profit_per_share": -0.2,
                },
                {
                    "date": "2026-01-02",
                    "market_ticker": "B",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 22,
                    "edge": 0.09,
                    "calibrated_expected_roi": 1.6,
                    "volume": 100,
                    "clv_cents": -2,
                    "realized_profit_per_share": -0.1,
                },
                {
                    "date": "2026-01-03",
                    "market_ticker": "C",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 60,
                    "edge": 0.02,
                    "calibrated_expected_roi": 0.2,
                    "volume": 500,
                    "clv_cents": 4,
                    "realized_profit_per_share": 0.3,
                },
                {
                    "date": "2026-01-04",
                    "market_ticker": "D",
                    "calibrated_trade": False,
                    "calibrated_side": "YES",
                    "price_cents": 20,
                    "edge": 0.08,
                    "calibrated_expected_roi": 1.5,
                    "volume": 100,
                    "clv_cents": 20,
                    "realized_profit_per_share": 0.8,
                },
            ]
        )

        reports, summary = build_edge_failure_diagnosis(rows, min_segment_rows=1)

        self.assertEqual(summary["signals"], 3)
        self.assertEqual(summary["status"], "not_proven")
        self.assertIn("worst_segments", reports)
        self.assertFalse(reports["worst_segments"].empty)
        self.assertIn("by_side_price", reports)
        self.assertGreater(summary["yes_signals"], 0)


if __name__ == "__main__":
    unittest.main()
