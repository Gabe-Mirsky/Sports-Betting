from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.stability import summarize_signal_stability  # noqa: E402


class TestSignalStability(unittest.TestCase):
    def test_summarize_signal_stability_groups_by_month(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "consensus_trade": True,
                    "actual_yes_win": True,
                    "realized_profit_per_share": 0.40,
                    "edge": 0.05,
                    "consensus_expected_roi": 0.10,
                },
                {
                    "date": "2025-01-02",
                    "consensus_trade": False,
                    "actual_yes_win": False,
                    "realized_profit_per_share": -0.50,
                    "edge": 0.02,
                    "consensus_expected_roi": 0.04,
                },
                {
                    "date": "2025-02-01",
                    "consensus_trade": True,
                    "actual_yes_win": False,
                    "realized_profit_per_share": -0.30,
                    "edge": 0.07,
                    "consensus_expected_roi": 0.12,
                },
            ]
        )

        monthly, summary = summarize_signal_stability(
            rows,
            signal_column="consensus_trade",
            expected_roi_column="consensus_expected_roi",
        )

        self.assertEqual(monthly["month"].tolist(), ["2025-01", "2025-02"])
        self.assertEqual(summary["signals"], 2)
        self.assertEqual(summary["positive_months"], 1)

    def test_summarize_signal_stability_uses_contract_outcome_for_no_side(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "calibrated_trade": True,
                    "candidate_side": "NO",
                    "actual_yes_win": False,
                    "contract_cost": 0.40,
                    "edge": 0.05,
                }
            ]
        )

        monthly, summary = summarize_signal_stability(rows, signal_column="calibrated_trade")

        self.assertAlmostEqual(float(monthly.loc[0, "win_rate"]), 1.0)
        self.assertAlmostEqual(float(monthly.loc[0, "avg_profit_per_share"]), 0.60)
        self.assertAlmostEqual(float(summary["overall_win_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()
