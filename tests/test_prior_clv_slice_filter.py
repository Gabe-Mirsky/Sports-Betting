from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.prior_clv_slice_filter import run_prior_clv_slice_filter  # noqa: E402


class TestPriorClvSliceFilter(unittest.TestCase):
    def test_filter_uses_prior_month_groups(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 20,
                    "edge": 0.06,
                    "volume": 100,
                    "clv_cents": 2,
                    "realized_profit_per_share": 0.2,
                },
                {
                    "date": "2026-02-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 22,
                    "edge": 0.07,
                    "volume": 120,
                    "clv_cents": 1,
                    "realized_profit_per_share": 0.1,
                },
                {
                    "date": "2026-03-01",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "price_cents": 23,
                    "edge": 0.07,
                    "volume": 150,
                    "clv_cents": 3,
                    "realized_profit_per_share": 0.3,
                },
                {
                    "date": "2026-03-02",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 65,
                    "edge": 0.07,
                    "volume": 150,
                    "clv_cents": -2,
                    "realized_profit_per_share": -0.2,
                },
            ]
        )

        selected, policies, folds, summary = run_prior_clv_slice_filter(
            rows,
            min_train_rows_values=[2],
            min_positive_clv_values=[0.5],
            min_train_months=2,
        )

        self.assertFalse(policies.empty)
        self.assertFalse(folds.empty)
        self.assertEqual(summary["policies_tested"], len(policies))
        self.assertGreaterEqual(len(selected), 1)
        self.assertIn("prior_clv_policy", selected.columns)


if __name__ == "__main__":
    unittest.main()
