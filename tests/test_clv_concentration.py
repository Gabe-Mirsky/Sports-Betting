from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.clv_concentration import run_clv_price_month_sweep, run_walk_forward_clv_price_month_validation  # noqa: E402


class TestCLVConcentration(unittest.TestCase):
    def test_price_month_sweep_finds_stable_price_range(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=120, freq="3D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "market_ticker": f"M{index}",
                    "clv_filtered_trade": True,
                    "clv_filter_side": "YES",
                    "price_cents": 30,
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.10,
                }
            )
        frame = pd.DataFrame(rows)

        rules, monthly, summary = run_clv_price_month_sweep(
            frame,
            price_breaks=[0, 25, 40, 55],
            min_rows=25,
        )

        self.assertFalse(rules.empty)
        self.assertFalse(monthly.empty)
        self.assertEqual(summary["best_status"], "stability_candidate")
        self.assertGreaterEqual(summary["best_rule_positive_clv_rate"], 0.50)

    def test_price_month_sweep_rejects_unstable_range(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=60, freq="3D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "market_ticker": f"M{index}",
                    "clv_filtered_trade": True,
                    "clv_filter_side": "YES",
                    "price_cents": 30,
                    "clv_cents": -1.0 if index % 2 else 1.0,
                    "realized_profit_per_share": -0.05,
                }
            )
        frame = pd.DataFrame(rows)

        rules, _, summary = run_clv_price_month_sweep(
            frame,
            price_breaks=[0, 25, 40],
            min_rows=25,
        )

        self.assertEqual(rules.loc[0, "status"], "not_ready")
        self.assertEqual(summary["best_status"], "not_ready")

    def test_walk_forward_validation_uses_prior_month_rule(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=150, freq="2D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "market_ticker": f"M{index}",
                    "clv_filtered_trade": True,
                    "clv_filter_side": "YES",
                    "price_cents": 30,
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.10,
                }
            )
        frame = pd.DataFrame(rows)

        validated, folds, monthly, summary = run_walk_forward_clv_price_month_validation(
            frame,
            price_breaks=[0, 25, 40, 55],
            min_rows=10,
            min_train_months=1,
        )

        self.assertFalse(validated.empty)
        self.assertIn("walk_forward_clv_price_signal", validated.columns)
        self.assertGreater(int(folds["status"].eq("evaluated").sum()), 0)
        self.assertFalse(monthly.empty)
        self.assertEqual(summary["status"], "walk_forward_candidate")

    def test_walk_forward_validation_skips_until_train_months_exist(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=30, freq="2D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "market_ticker": f"M{index}",
                    "clv_filtered_trade": True,
                    "clv_filter_side": "YES",
                    "price_cents": 30,
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.10,
                }
            )
        frame = pd.DataFrame(rows)

        _, folds, _, summary = run_walk_forward_clv_price_month_validation(
            frame,
            price_breaks=[0, 25, 40],
            min_rows=10,
            min_train_months=10,
        )

        self.assertTrue(folds["status"].astype(str).str.startswith("skipped").all())
        self.assertEqual(summary["signals"], 0)


if __name__ == "__main__":
    unittest.main()
