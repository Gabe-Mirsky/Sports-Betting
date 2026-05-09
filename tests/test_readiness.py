from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.readiness import evaluate_strategy_readiness  # noqa: E402


class TestReadiness(unittest.TestCase):
    def test_readiness_marks_profitable_but_unstable_as_watchlist(self) -> None:
        row = evaluate_strategy_readiness(
            "candidate",
            monthly=pd.DataFrame(),
            stability_summary={
                "signals": 150,
                "months": 6,
                "positive_month_share": 0.50,
                "overall_avg_profit_per_share": 0.02,
                "timeline": "2025-01-01 to 2025-06-01",
            },
            portfolio_summary={
                "ending_bankroll": 105.0,
                "total_return_pct": 0.05,
                "max_drawdown": -0.20,
                "trade_timeline": "2025-01-01 to 2025-06-01",
            },
        )

        self.assertEqual(row["status"], "watchlist")
        self.assertIn("unstable_monthly_profit", row["failed_checks"])
        self.assertFalse(row["parlay_ready"])

    def test_readiness_rejects_losing_portfolio(self) -> None:
        row = evaluate_strategy_readiness(
            "loser",
            monthly=pd.DataFrame(),
            stability_summary={
                "signals": 150,
                "months": 6,
                "positive_month_share": 0.80,
                "overall_avg_profit_per_share": 0.02,
            },
            portfolio_summary={
                "ending_bankroll": 90.0,
                "total_return_pct": -0.10,
                "max_drawdown": -0.20,
            },
        )

        self.assertEqual(row["status"], "not_ready")
        self.assertIn("portfolio_lost_money", row["failed_checks"])


if __name__ == "__main__":
    unittest.main()
