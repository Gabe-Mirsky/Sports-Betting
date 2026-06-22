from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.no_shrinkage import run_no_shrinkage_research  # noqa: E402


class TestNoShrinkage(unittest.TestCase):
    def test_shrinkage_reduces_adjusted_no_probability(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 20,
                    "contract_cost": 0.20,
                    "calibrated_win_rate": 0.35,
                    "clv_cents": 2,
                    "realized_profit_per_share": 0.80,
                    "actual_contract_win": True,
                },
                {
                    "date": "2026-01-02",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 45,
                    "contract_cost": 0.45,
                    "calibrated_win_rate": 0.48,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.45,
                    "actual_contract_win": False,
                },
                {
                    "date": "2026-02-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 22,
                    "contract_cost": 0.22,
                    "calibrated_win_rate": 0.36,
                    "clv_cents": 1,
                    "realized_profit_per_share": 0.78,
                    "actual_contract_win": True,
                },
                {
                    "date": "2026-03-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 23,
                    "contract_cost": 0.23,
                    "calibrated_win_rate": 0.37,
                    "clv_cents": 1,
                    "realized_profit_per_share": -0.23,
                    "actual_contract_win": False,
                },
            ]
        )

        descriptive, validated, folds, summary = run_no_shrinkage_research(
            rows,
            min_train_months=1,
            min_rows=1,
        )

        self.assertFalse(descriptive.empty)
        self.assertFalse(validated.empty)
        self.assertFalse(folds.empty)
        self.assertIn("no_shrink_adjusted_win_rate", validated.columns)
        selected = validated[validated["no_shrinkage_signal"].fillna(False)]
        self.assertTrue((selected["no_shrink_adjusted_win_rate"] <= selected["calibrated_win_rate"]).all())
        self.assertFalse(summary["single_game_edge_proven"])


if __name__ == "__main__":
    unittest.main()
