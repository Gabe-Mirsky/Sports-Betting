from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.no_calibration_guardrail import run_no_calibration_guardrail_research  # noqa: E402


class TestNoCalibrationGuardrail(unittest.TestCase):
    def test_guardrail_runs_walk_forward_without_promoting_to_proven(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 15,
                    "calibrated_win_rate": 0.25,
                    "clv_cents": 1,
                    "realized_profit_per_share": 0.85,
                    "actual_contract_win": True,
                },
                {
                    "date": "2026-01-02",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 45,
                    "calibrated_win_rate": 0.50,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.45,
                    "actual_contract_win": False,
                },
                {
                    "date": "2026-02-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 16,
                    "calibrated_win_rate": 0.26,
                    "clv_cents": 2,
                    "realized_profit_per_share": 0.84,
                    "actual_contract_win": True,
                },
                {
                    "date": "2026-03-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 17,
                    "calibrated_win_rate": 0.27,
                    "clv_cents": 1,
                    "realized_profit_per_share": -0.17,
                    "actual_contract_win": False,
                },
            ]
        )

        descriptive, validated, folds, summary = run_no_calibration_guardrail_research(
            rows,
            min_train_months=1,
            min_rows=1,
        )

        self.assertFalse(descriptive.empty)
        self.assertFalse(validated.empty)
        self.assertFalse(folds.empty)
        self.assertIn("descriptive_best_policy", summary)
        self.assertFalse(summary["single_game_edge_proven"])

    def test_guardrail_can_use_player_edge_higher_policy(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 25,
                    "calibrated_win_rate": 0.30,
                    "clv_cents": 1,
                    "realized_profit_per_share": 0.75,
                    "actual_contract_win": True,
                    "player_edge_higher": True,
                    "same_side": True,
                },
                {
                    "date": "2026-01-02",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 25,
                    "calibrated_win_rate": 0.30,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.25,
                    "actual_contract_win": False,
                    "player_edge_higher": False,
                    "same_side": True,
                },
            ]
        )

        descriptive, _, _, _ = run_no_calibration_guardrail_research(
            rows,
            min_train_months=1,
            min_rows=1,
        )

        policies = set(descriptive["policy"])
        self.assertIn("no_player_edge_higher", policies)
        selected = descriptive.set_index("policy").loc["no_player_edge_higher"]
        self.assertEqual(int(selected["signals"]), 1)


if __name__ == "__main__":
    unittest.main()
