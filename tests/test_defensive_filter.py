from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.defensive_filter import (  # noqa: E402
    add_defensive_filters,
    run_defensive_rule_sweep,
    run_defensive_sample_expansion,
    run_walk_forward_defensive_validation,
)


class TestDefensiveFilter(unittest.TestCase):
    def test_defensive_filter_blocks_decay_prone_slices(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "clv_filtered_trade": True,
                    "price_cents": 12,
                    "calibrated_expected_roi": 0.2,
                    "volume": 100,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
                {
                    "clv_filtered_trade": True,
                    "price_cents": 42,
                    "calibrated_expected_roi": 2.0,
                    "volume": 100,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
                {
                    "clv_filtered_trade": True,
                    "price_cents": 12,
                    "calibrated_expected_roi": 2.0,
                    "volume": 100,
                    "clv_cents": 3,
                    "realized_profit_per_share": 0.1,
                },
                {
                    "clv_filtered_trade": True,
                    "price_cents": 8,
                    "calibrated_expected_roi": 2.0,
                    "volume": 100,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
                {
                    "clv_filtered_trade": True,
                    "price_cents": 12,
                    "calibrated_expected_roi": 3.5,
                    "volume": 100,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
                {
                    "clv_filtered_trade": True,
                    "price_cents": 12,
                    "calibrated_expected_roi": 2.0,
                    "volume": 1200,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
            ]
        )

        filtered, audit, summary = add_defensive_filters(
            rows,
            min_calibrated_expected_roi=0.5,
            max_price_cents=40,
        )

        self.assertEqual(filtered["defensive_trade"].tolist(), [False, False, True, False, False, False])
        self.assertEqual(summary["defensive_trades"], 1)
        self.assertIn("calibrated_roi_below_minimum", set(audit["reason"]))
        self.assertIn("price_above_maximum", set(audit["reason"]))
        self.assertIn("price_below_minimum", set(audit["reason"]))
        self.assertIn("calibrated_roi_too_high", set(audit["reason"]))
        self.assertIn("volume_above_maximum", set(audit["reason"]))

    def test_defensive_filter_respects_base_signal(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "clv_filtered_trade": False,
                    "price_cents": 12,
                    "calibrated_expected_roi": 2.0,
                    "volume": 100,
                    "clv_cents": 3,
                    "realized_profit_per_share": 0.1,
                }
            ]
        )

        filtered, _, summary = add_defensive_filters(rows)

        self.assertFalse(bool(filtered.loc[0, "defensive_trade"]))
        self.assertEqual(summary["base_signals"], 0)

    def test_defensive_rule_sweep_scores_thresholds(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=80, freq="3D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "clv_filtered_trade": True,
                    "price_cents": 12,
                    "calibrated_expected_roi": 2.0,
                    "volume": 100,
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.1,
                }
            )
        frame = pd.DataFrame(rows)

        rules, monthly, summary = run_defensive_rule_sweep(
            frame,
            min_price_values=[10],
            max_price_values=[40],
            min_roi_values=[0.5],
            max_roi_values=[3],
            max_volume_values=[1000],
            min_rows=10,
        )

        self.assertFalse(rules.empty)
        self.assertFalse(monthly.empty)
        self.assertEqual(summary["best_status"], "watchlist")
        self.assertEqual(rules.loc[0, "max_price_cents"], 40)
        self.assertEqual(rules.loc[0, "min_calibrated_expected_roi"], 0.5)
        self.assertEqual(summary["best_rule_positive_clv_rate"], 1.0)

    def test_walk_forward_defensive_validation_uses_prior_months(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=150, freq="2D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "clv_filtered_trade": True,
                    "price_cents": 12,
                    "calibrated_expected_roi": 2.0,
                    "volume": 100,
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.1,
                }
            )
        frame = pd.DataFrame(rows)

        validated, folds, monthly, summary = run_walk_forward_defensive_validation(
            frame,
            min_price_values=[10],
            max_price_values=[40],
            min_roi_values=[0.5],
            max_roi_values=[3],
            max_volume_values=[1000],
            min_rows=10,
            min_train_months=1,
        )

        self.assertFalse(validated.empty)
        self.assertIn("walk_forward_defensive_signal", validated.columns)
        self.assertGreater(int(folds["status"].eq("evaluated").sum()), 0)
        self.assertFalse(monthly.empty)
        self.assertEqual(summary["status"], "walk_forward_candidate")

    def test_defensive_sample_expansion_finds_larger_clv_safe_rule(self) -> None:
        rows = []
        dates = pd.date_range("2025-01-01", periods=180, freq="2D")
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "clv_filtered_trade": True,
                    "price_cents": 35 if index % 3 else 45,
                    "calibrated_expected_roi": 1.0,
                    "volume": 500,
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.1,
                }
            )
        frame = pd.DataFrame(rows)

        candidates, monthly, summary = run_defensive_sample_expansion(
            frame,
            min_price_values=[10],
            max_price_values=[40, 55],
            min_roi_values=[0.5],
            max_roi_values=[3],
            max_volume_values=[1000],
            min_train_months=1,
            target_min_signals=100,
            target_max_signals=150,
        )

        self.assertFalse(candidates.empty)
        self.assertFalse(monthly.empty)
        self.assertEqual(summary["status"], "sample_expansion_candidate")
        self.assertGreaterEqual(summary["best_rule_signals"], 100)
        self.assertEqual(summary["best_rule_positive_clv_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
