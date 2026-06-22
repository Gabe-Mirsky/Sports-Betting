from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.player_edge_agreement import build_player_edge_agreement_report  # noqa: E402


class TestPlayerEdgeAgreement(unittest.TestCase):
    def test_agreement_report_builds_descriptive_and_folds(self) -> None:
        player_rows = []
        team_rows = []
        for month_index, month in enumerate(["2026-01", "2026-02", "2026-03"]):
            for item in range(12):
                game_id = f"g{month_index}-{item}"
                ticker = f"m{month_index}-{item}"
                player_rows.append(
                    {
                        "date": f"{month}-{item + 1:02d}",
                        "game_id": game_id,
                        "market_ticker": ticker,
                        "trade": True,
                        "candidate_side": "YES",
                        "edge": 0.08,
                        "profit": 0.10 if item % 2 == 0 else -0.05,
                        "clv_cents": 1.0 if item % 2 == 0 else -0.5,
                    }
                )
                team_rows.append(
                    {
                        "date": f"{month}-{item + 1:02d}",
                        "game_id": game_id,
                        "market_ticker": ticker,
                        "trade": item < 10,
                        "candidate_side": "YES" if item < 10 else "NO",
                        "edge": 0.06,
                    }
                )
        rows, descriptive, folds, summary = build_player_edge_agreement_report(
            pd.DataFrame(player_rows),
            pd.DataFrame(team_rows),
            min_train_months=1,
        )

        self.assertFalse(rows.empty)
        self.assertIn("same_side", set(descriptive["policy"]))
        self.assertTrue(folds["status"].isin(["evaluated", "skipped_insufficient_prior_months"]).all())
        self.assertGreaterEqual(summary["signals"], 0)

    def test_agreement_report_handles_no_rows(self) -> None:
        rows, descriptive, folds, summary = build_player_edge_agreement_report(
            pd.DataFrame(),
            pd.DataFrame(),
        )

        self.assertTrue(rows.empty)
        self.assertTrue(descriptive.empty)
        self.assertTrue(folds.empty)
        self.assertEqual(summary["status"], "no_rows")

    def test_agreement_report_accepts_custom_signal_column(self) -> None:
        player = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "game_id": "g1",
                    "market_ticker": "m1",
                    "calibrated_trade": True,
                    "candidate_side": "NO",
                    "edge": 0.08,
                    "realized_profit_per_share": 0.10,
                    "clv_cents": 1.0,
                }
            ]
        )
        team = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "game_id": "g1",
                    "market_ticker": "m1",
                    "trade": True,
                    "candidate_side": "NO",
                    "edge": 0.04,
                }
            ]
        )

        rows, descriptive, _, summary = build_player_edge_agreement_report(
            player,
            team,
            player_signal_column="calibrated_trade",
        )

        self.assertTrue(bool(rows.loc[0, "trade"]))
        self.assertEqual(float(rows.loc[0, "profit"]), 0.10)
        self.assertFalse(descriptive.empty)
        self.assertEqual(summary["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
