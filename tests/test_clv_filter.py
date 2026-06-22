from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.clv_filter import add_expanding_clv_filter  # noqa: E402


class TestCLVFilter(unittest.TestCase):
    def test_clv_filter_uses_prior_side_history_only(self) -> None:
        rows = []
        for index in range(4):
            rows.append(
                {
                    "date": f"2025-01-{index + 1:02d}",
                    "market_ticker": f"Y{index}",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "clv_cents": 2.0,
                    "realized_profit_per_share": 0.10,
                }
            )
        frame = pd.DataFrame(rows)

        filtered, side_audit, summary = add_expanding_clv_filter(
            frame,
            side_rules={
                "YES": {
                    "min_history_rows": 2,
                    "min_avg_clv_cents": 0.0,
                    "min_positive_clv_rate": 0.50,
                    "min_avg_profit_per_share": 0.0,
                },
                "NO": {
                    "min_history_rows": 2,
                    "min_avg_clv_cents": 0.0,
                    "min_positive_clv_rate": 0.50,
                    "min_avg_profit_per_share": 0.0,
                },
            },
        )

        self.assertEqual(filtered["clv_filtered_trade"].tolist(), [False, False, True, True])
        self.assertEqual(summary["clv_filtered_trades"], 2)
        self.assertEqual(int(side_audit.loc[side_audit["side"].eq("YES"), "clv_filtered_trades"].iloc[0]), 2)

    def test_clv_filter_blocks_no_side_with_bad_prior_clv(self) -> None:
        rows = []
        for index in range(4):
            rows.append(
                {
                    "date": f"2025-01-{index + 1:02d}",
                    "market_ticker": f"N{index}",
                    "calibrated_trade": True,
                    "calibrated_side": "NO",
                    "clv_cents": -1.0,
                    "realized_profit_per_share": -0.10,
                }
            )
        frame = pd.DataFrame(rows)

        filtered, _, summary = add_expanding_clv_filter(
            frame,
            side_rules={
                "YES": {
                    "min_history_rows": 1,
                    "min_avg_clv_cents": 0.0,
                    "min_positive_clv_rate": 0.0,
                    "min_avg_profit_per_share": 0.0,
                },
                "NO": {
                    "min_history_rows": 1,
                    "min_avg_clv_cents": 0.0,
                    "min_positive_clv_rate": 0.50,
                    "min_avg_profit_per_share": 0.0,
                },
            },
        )

        self.assertFalse(bool(filtered["clv_filtered_trade"].any()))
        self.assertEqual(summary["no_clv_filtered_trades"], 0)


if __name__ == "__main__":
    unittest.main()
