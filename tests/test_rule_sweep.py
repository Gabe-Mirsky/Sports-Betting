from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.rule_sweep import (  # noqa: E402
    build_rule_signal,
    run_signal_rule_sweep,
    run_walk_forward_signal_rule_validation,
)


class TestSignalRuleSweep(unittest.TestCase):
    def test_build_rule_signal_applies_all_filters(self) -> None:
        rows = pd.DataFrame(
            {
                "consensus_trade": [True, True, True, False],
                "consensus_expected_roi": [0.40, 0.60, 0.80, 0.90],
                "edge": [0.04, 0.06, 0.08, 0.10],
                "price_cents": [50, 55, 96, 45],
                "edge_bin_history_rows": [120, 80, 200, 200],
                "edge_bin_history_rows_blend": [130, 130, 220, 220],
            }
        )

        signal = build_rule_signal(
            rows,
            signal_column="consensus_trade",
            expected_roi_column="consensus_expected_roi",
            min_edge=0.05,
            min_expected_roi=0.50,
            min_history_rows=100,
            min_price_cents=10,
            max_price_cents=90,
        )

        self.assertEqual(signal.tolist(), [False, False, False, False])

    def test_run_signal_rule_sweep_sorts_candidate_rules_first(self) -> None:
        rows = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=120, freq="3D"),
                "consensus_trade": [True] * 120,
                "consensus_expected_roi": [0.60] * 120,
                "edge": [0.06] * 120,
                "price_cents": [40] * 120,
                "edge_bin_history_rows": [150] * 120,
                "edge_bin_history_rows_blend": [150] * 120,
                "actual_yes_win": [True, True, False, True] * 30,
                "realized_profit_per_share": [0.60, 0.60, -0.40, 0.60] * 30,
            }
        )

        rules, monthly, summary = run_signal_rule_sweep(
            rows,
            min_edges=[0.05],
            min_expected_rois=[0.50],
            min_history_rows=[100],
            min_price_cents=[10],
            max_price_cents=[90],
        )

        self.assertEqual(len(rules), 1)
        self.assertEqual(rules.loc[0, "status"], "exploratory_candidate")
        self.assertEqual(rules.loc[0, "signals"], 120)
        self.assertFalse(bool(rules.loc[0, "parlay_ready"]))
        self.assertFalse(monthly.empty)
        self.assertEqual(summary["best_rule_signals"], 120)
        self.assertIn("paper-watch", summary["note"])

    def test_walk_forward_validation_selects_rules_from_prior_months(self) -> None:
        rows = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=90, freq="2D"),
                "consensus_trade": [True] * 90,
                "consensus_expected_roi": [0.60] * 90,
                "edge": [0.07] * 90,
                "price_cents": [40] * 90,
                "edge_bin_history_rows": [150] * 90,
                "edge_bin_history_rows_blend": [150] * 90,
                "actual_yes_win": [True, True, False] * 30,
                "realized_profit_per_share": [0.60, 0.60, -0.40] * 30,
            }
        )

        validated, folds, monthly, summary = run_walk_forward_signal_rule_validation(
            rows,
            min_edges=[0.05],
            min_expected_rois=[0.50],
            min_history_rows=[100],
            min_price_cents=[10],
            max_price_cents=[90],
            min_train_rows=20,
            min_train_months=1,
        )

        self.assertFalse(validated.empty)
        self.assertIn("walk_forward_rule_signal", validated.columns)
        self.assertGreater(int(folds["status"].eq("evaluated").sum()), 0)
        self.assertFalse(monthly.empty)
        self.assertGreater(summary["evaluated_months"], 0)
        self.assertFalse(summary["parlay_ready"])


if __name__ == "__main__":
    unittest.main()
