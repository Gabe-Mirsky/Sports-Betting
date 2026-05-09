from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.diagnostics import (  # noqa: E402
    backtest_edge_bins,
    prediction_probability_bins,
    season_prediction_summary,
    top_backtest_trades,
)


class TestDiagnostics(unittest.TestCase):
    def test_prediction_bins_compare_predicted_to_observed_rates(self) -> None:
        predictions = pd.DataFrame(
            {
                "model_home_win_prob": [0.10, 0.20, 0.80, 0.90],
                "actual_home_win": [0, 1, 1, 1],
            }
        )

        bins = prediction_probability_bins(predictions, bins=2)

        self.assertEqual(int(bins["games"].sum()), 4)
        low_bin = bins.iloc[0]
        self.assertAlmostEqual(float(low_bin["avg_predicted_prob"]), 0.15)
        self.assertAlmostEqual(float(low_bin["observed_win_rate"]), 0.50)

    def test_season_summary_is_time_grouped(self) -> None:
        predictions = pd.DataFrame(
            {
                "season": [2023, 2023, 2024, 2024],
                "model_home_win_prob": [0.60, 0.40, 0.70, 0.20],
                "actual_home_win": [1, 0, 0, 0],
            }
        )

        summary = season_prediction_summary(predictions)

        self.assertEqual(summary["season"].tolist(), [2023, 2024])
        self.assertEqual(summary["games"].tolist(), [2, 2])
        self.assertAlmostEqual(float(summary.loc[summary["season"] == 2023, "accuracy"].iloc[0]), 1.0)

    def test_edge_bins_handle_saved_boolean_strings(self) -> None:
        trades = pd.DataFrame(
            {
                "edge": [0.06, 0.07, 0.01],
                "trade": ["True", "False", "False"],
                "actual_yes_win": ["True", "False", "True"],
                "profit": [0.50, 0.0, 0.0],
                "cost": [0.50, 0.0, 0.0],
            }
        )

        bins = backtest_edge_bins(trades, bins=[0.0, 0.05, 0.10])

        high_bin = bins.iloc[1]
        self.assertEqual(int(high_bin["markets"]), 2)
        self.assertEqual(int(high_bin["trades"]), 1)
        self.assertAlmostEqual(float(high_bin["traded_win_rate"]), 1.0)
        self.assertAlmostEqual(float(high_bin["total_profit"]), 0.50)

    def test_top_backtest_trades_returns_largest_absolute_profit(self) -> None:
        trades = pd.DataFrame(
            {
                "game_id": ["0022400061", "0022400062", "0022400063"],
                "trade": [True, True, False],
                "profit": [2.0, -3.0, 9.0],
                "edge": [0.10, 0.08, 0.20],
            }
        )

        top = top_backtest_trades(trades, n=2)

        self.assertEqual(top["profit"].tolist(), [-3.0, 2.0])
        self.assertEqual(top["result_type"].tolist(), ["loss", "win"])
        self.assertEqual(top["game_id"].tolist(), ["0022400062", "0022400061"])


if __name__ == "__main__":
    unittest.main()
