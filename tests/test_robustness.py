from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.robustness import add_confidence_screen  # noqa: E402


class TestRobustness(unittest.TestCase):
    def test_confidence_screen_requires_lower_bound_profit(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "consensus_trade": True,
                    "consensus_expected_roi": 0.20,
                    "calibrated_yes_rate": 0.70,
                    "edge_bin_history_rows": 200,
                    "contract_cost": 0.50,
                    "actual_yes_win": True,
                    "realized_profit_per_share": 0.50,
                },
                {
                    "date": "2025-01-02",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "consensus_trade": True,
                    "consensus_expected_roi": 0.02,
                    "calibrated_yes_rate": 0.53,
                    "edge_bin_history_rows": 20,
                    "contract_cost": 0.52,
                    "actual_yes_win": False,
                    "realized_profit_per_share": -0.52,
                },
            ]
        )

        screened, summary = add_confidence_screen(
            rows,
            signal_column="consensus_trade",
            expected_roi_column="consensus_expected_roi",
            min_history_rows=100,
            confidence_z=0.75,
        )

        self.assertEqual(screened["robust_calibrated_trade"].tolist(), [True, False])
        self.assertEqual(summary["robust_signals"], 1)
        self.assertEqual(screened.loc[1, "robust_reason"], "insufficient_confidence_history")

    def test_confidence_screen_uses_blend_lower_bound_when_available(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "consensus_trade": True,
                    "consensus_expected_roi": 0.20,
                    "calibrated_yes_rate": 0.70,
                    "edge_bin_history_rows": 200,
                    "calibrated_yes_rate_blend": 0.51,
                    "edge_bin_history_rows_blend": 200,
                    "contract_cost": 0.50,
                    "actual_yes_win": True,
                    "realized_profit_per_share": 0.50,
                }
            ]
        )

        screened, summary = add_confidence_screen(
            rows,
            signal_column="consensus_trade",
            expected_roi_column="consensus_expected_roi",
            blend_probability_column="calibrated_yes_rate_blend",
            blend_sample_size_column="edge_bin_history_rows_blend",
            min_history_rows=100,
            confidence_z=0.75,
        )

        self.assertFalse(bool(screened.loc[0, "robust_calibrated_trade"]))
        self.assertEqual(summary["confidence_source"], "raw_and_market_blend_lower_bound")


if __name__ == "__main__":
    unittest.main()
