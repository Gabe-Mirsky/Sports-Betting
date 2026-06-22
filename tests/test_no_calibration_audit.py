from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.no_calibration_audit import build_no_calibration_audit  # noqa: E402


class TestNoCalibrationAudit(unittest.TestCase):
    def test_no_calibration_audit_reports_overconfident_no_slice(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "market_ticker": "A",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 25,
                    "clv_cents": 1,
                    "actual_contract_win": False,
                    "realized_profit_per_share": -0.25,
                    "calibrated_win_rate": 0.40,
                    "edge": 0.05,
                    "volume": 500,
                },
                {
                    "date": "2026-01-02",
                    "market_ticker": "B",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 30,
                    "clv_cents": 2,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.70,
                    "calibrated_win_rate": 0.45,
                    "edge": 0.06,
                    "volume": 500,
                },
                {
                    "date": "2026-01-03",
                    "market_ticker": "C",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 30,
                    "clv_cents": 10,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.70,
                    "calibrated_win_rate": 0.80,
                    "edge": 0.06,
                    "volume": 500,
                },
            ]
        )

        reports, summary = build_no_calibration_audit(rows, min_segment_rows=1)

        self.assertEqual(summary["selected_no_rows"], 2)
        self.assertAlmostEqual(summary["avg_forecast_win_rate"], 0.425)
        self.assertAlmostEqual(summary["actual_win_rate"], 0.5)
        self.assertIn("by_forecast_win_bucket", reports)
        self.assertIn("by_forecast_price", reports)
        self.assertEqual(len(reports["positive_clv_losses"]), 1)
        self.assertFalse(summary["single_game_edge_proven"])


if __name__ == "__main__":
    unittest.main()
