from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.side_suppression import run_side_suppression_research  # noqa: E402


class TestSideSuppression(unittest.TestCase):
    def test_side_suppression_can_select_no_only_policy(self) -> None:
        rows = []
        for index, date in enumerate(pd.date_range("2026-01-01", periods=90, freq="D")):
            side = "NO" if index % 2 else "YES"
            rows.append(
                {
                    "date": date,
                    "market_ticker": f"M{index}",
                    "calibrated_trade": True,
                    "calibrated_side": side,
                    "price_cents": 40,
                    "edge": 0.10,
                    "clv_cents": 2.0 if side == "NO" else -2.0,
                    "realized_profit_per_share": 0.20 if side == "NO" else -0.20,
                }
            )

        descriptive, validated, folds, summary = run_side_suppression_research(
            pd.DataFrame(rows),
            min_train_months=1,
            min_rows=5,
        )

        self.assertFalse(descriptive.empty)
        self.assertIn("no_only", set(descriptive["policy"]))
        self.assertFalse(validated.empty)
        self.assertFalse(folds.empty)
        self.assertGreaterEqual(summary["signals"], 1)
        self.assertFalse(summary["single_game_edge_proven"])


if __name__ == "__main__":
    unittest.main()
