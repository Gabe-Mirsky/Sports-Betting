from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.clv_decay import build_clv_decay_audit  # noqa: E402


class TestCLVDecay(unittest.TestCase):
    def test_build_clv_decay_audit_detects_monthly_decay(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-02-01",
                    "market_ticker": "A",
                    "walk_forward_clv_price_signal": True,
                    "price_cents": 20,
                    "edge": 0.05,
                    "calibrated_expected_roi": 1.0,
                    "volume": 100,
                    "clv_cents": 3,
                    "realized_profit_per_share": 0.1,
                },
                {
                    "date": "2026-02-02",
                    "market_ticker": "B",
                    "walk_forward_clv_price_signal": True,
                    "price_cents": 20,
                    "edge": 0.05,
                    "calibrated_expected_roi": 1.0,
                    "volume": 100,
                    "clv_cents": 2,
                    "realized_profit_per_share": 0.1,
                },
                {
                    "date": "2026-04-01",
                    "market_ticker": "C",
                    "walk_forward_clv_price_signal": True,
                    "price_cents": 20,
                    "edge": 0.05,
                    "calibrated_expected_roi": 1.0,
                    "volume": 100,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
                {
                    "date": "2026-04-02",
                    "market_ticker": "D",
                    "walk_forward_clv_price_signal": False,
                    "price_cents": 20,
                    "edge": 0.05,
                    "calibrated_expected_roi": 1.0,
                    "volume": 100,
                    "clv_cents": 5,
                    "realized_profit_per_share": 0.1,
                },
            ]
        )

        reports, summary = build_clv_decay_audit(rows)

        self.assertEqual(summary["status"], "decay_detected")
        self.assertLess(summary["positive_clv_rate_change"], 0)
        self.assertIn("monthly", reports)
        self.assertIn("decay_drivers", reports)
        self.assertEqual(int(reports["negative_clv_rows"].shape[0]), 1)


if __name__ == "__main__":
    unittest.main()
