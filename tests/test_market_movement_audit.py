from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.market_movement_audit import build_market_movement_audit  # noqa: E402


class TestMarketMovementAudit(unittest.TestCase):
    def test_market_movement_audit_splits_profit_from_clv(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 40,
                    "clv_reference_price_cents": 38,
                    "clv_cents": -2,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.60,
                    "edge": 0.05,
                    "calibrated_expected_roi": 0.2,
                    "volume": 500,
                    "clv_reference_snapshot": "pregame_5m",
                },
                {
                    "date": "2026-01-02",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 20,
                    "clv_reference_price_cents": 25,
                    "clv_cents": 5,
                    "actual_contract_win": False,
                    "realized_profit_per_share": -0.20,
                    "edge": -0.05,
                    "calibrated_expected_roi": 0.5,
                    "volume": 50,
                    "clv_reference_snapshot": "pregame_5m",
                },
            ]
        )

        reports, summary = build_market_movement_audit(rows)

        self.assertEqual(summary["signals"], 2)
        self.assertEqual(summary["profit_without_clv_count"], 1)
        self.assertEqual(summary["clv_without_profit_count"], 1)
        self.assertIn("by_side_move_outcome", reports)
        self.assertFalse(reports["profit_without_clv"].empty)


if __name__ == "__main__":
    unittest.main()
