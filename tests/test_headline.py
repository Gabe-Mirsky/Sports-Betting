from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.headline import build_headline_backtest_summary  # noqa: E402


class TestHeadlineBacktest(unittest.TestCase):
    def test_prefers_consensus_slate_and_carries_readiness(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_headline"
        root.mkdir(parents=True, exist_ok=True)
        try:
            (root / "portfolio_summary_consensus_calibrated.json").write_text(
                json.dumps(
                    {
                        "starting_bankroll": 100.0,
                        "ending_bankroll": 112.0,
                        "total_return_pct": 0.12,
                        "num_selected_trades": 20,
                        "num_slates": 8,
                        "trade_timeline": "2026-01-01 to 2026-02-01",
                    }
                ),
                encoding="utf-8",
            )
            (root / "portfolio_summary_calibrated.json").write_text(
                json.dumps({"ending_bankroll": 200.0, "num_selected_trades": 99}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "strategy": "consensus_calibrated",
                        "status": "watchlist",
                        "parlay_ready": False,
                        "failed_checks": "unstable_monthly_profit",
                    }
                ]
            ).to_csv(root / "strategy_readiness.csv", index=False)

            summary = build_headline_backtest_summary(root)
        finally:
            for child in root.glob("*"):
                child.unlink()
            root.rmdir()

        self.assertEqual(summary["headline_strategy"], "consensus_calibrated")
        self.assertEqual(summary["settlement_mode"], "slate_settled")
        self.assertEqual(summary["num_selected_trades"], 20)
        self.assertEqual(summary["readiness_status"], "watchlist")
        self.assertTrue(summary["parlays_blocked"])


if __name__ == "__main__":
    unittest.main()
