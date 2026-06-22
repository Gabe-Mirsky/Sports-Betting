from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.no_side_audit import build_no_side_audit  # noqa: E402


class TestNoSideAudit(unittest.TestCase):
    def test_no_side_audit_flags_positive_clv_losses_without_math_mismatch(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-02-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "candidate_side": "NO",
                    "clv_filtered_trade": True,
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "price_cents": 20,
                    "clv_reference_price_cents": 45,
                    "clv_cents": 25,
                    "actual_contract_win": False,
                    "realized_profit_per_share": -0.2,
                    "calibrated_expected_roi": 2.0,
                    "edge": 0.1,
                    "volume": 50,
                    "open_interest": 500,
                    "clv_reference_snapshot": "pregame_5m",
                },
                {
                    "date": "2026-02-02",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "candidate_side": "YES",
                    "clv_filtered_trade": True,
                    "price_cents": 20,
                    "clv_reference_price_cents": 30,
                    "clv_cents": 10,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.8,
                },
            ]
        )

        reports, summary = build_no_side_audit(rows)

        self.assertEqual(summary["selected_no_rows"], 1)
        self.assertEqual(summary["positive_clv_loss_count"], 1)
        self.assertEqual(summary["large_positive_clv_loss_count"], 1)
        self.assertEqual(summary["profit_math_mismatch_count"], 0)
        self.assertIn("positive_clv_losses", reports)
        self.assertFalse(reports["monthly"].empty)

    def test_no_side_audit_detects_profit_math_mismatch(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-02-01",
                    "clv_filter_side": "NO",
                    "clv_filtered_trade": True,
                    "price_cents": 20,
                    "clv_reference_price_cents": 30,
                    "clv_cents": 10,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.1,
                }
            ]
        )

        _, summary = build_no_side_audit(rows)

        self.assertEqual(summary["profit_math_mismatch_count"], 1)

    def test_no_side_audit_supports_calibrated_side_regime_buckets(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-02-01",
                    "market_ticker": "N1",
                    "calibrated_side": "NO",
                    "calibrated_trade": True,
                    "price_cents": 35,
                    "yes_bid": 64,
                    "yes_ask": 66,
                    "clv_reference_price_cents": 36,
                    "clv_cents": 1,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.65,
                    "calibrated_expected_roi": 0.8,
                    "edge": 0.04,
                    "volume": 500,
                    "open_interest": 1000,
                    "clv_reference_snapshot": "pregame_5m",
                }
            ]
        )

        reports, summary = build_no_side_audit(rows, signal_column="calibrated_trade")

        self.assertEqual(summary["selected_no_rows"], 1)
        self.assertIn("by_yes_market_price_bucket", reports)
        self.assertIn("by_spread_bucket", reports)
        self.assertIn("by_edge_bucket", reports)
        self.assertEqual(summary["wide_spread_count"], 0)


if __name__ == "__main__":
    unittest.main()
