from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.parlay_research import (  # noqa: E402
    apply_strategy_readiness_gate,
    build_parlay_pair_frame,
    summarize_parlay_pairs,
)


class TestParlayResearch(unittest.TestCase):
    def test_build_parlay_pair_frame_creates_same_slate_pairs(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "game_id": "g1",
                    "market_ticker": "m1",
                    "consensus_trade": True,
                    "actual_yes_win": True,
                    "yes_team_abbr": "BOS",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "market_prob": 0.55,
                    "consensus_expected_profit_per_share": 0.10,
                },
                {
                    "date": "2026-01-01",
                    "game_id": "g2",
                    "market_ticker": "m2",
                    "consensus_trade": True,
                    "actual_yes_win": False,
                    "yes_team_abbr": "LAL",
                    "home_team_abbr": "DEN",
                    "away_team_abbr": "LAL",
                    "market_prob": 0.45,
                    "consensus_expected_profit_per_share": 0.05,
                },
                {
                    "date": "2026-01-02",
                    "game_id": "g3",
                    "market_ticker": "m3",
                    "consensus_trade": False,
                    "actual_yes_win": True,
                    "yes_team_abbr": "MIA",
                    "home_team_abbr": "MIA",
                    "away_team_abbr": "ORL",
                    "market_prob": 0.60,
                    "consensus_expected_profit_per_share": 0.10,
                },
            ]
        )

        pairs = build_parlay_pair_frame(rows)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.loc[0, "side_combo"], "away_home")
        self.assertFalse(bool(pairs.loc[0, "pair_win"]))

    def test_summarize_parlay_pairs_blocks_low_sample(self) -> None:
        pairs = pd.DataFrame(
            {
                "date": ["2026-01-01"],
                "actual_yes_win_1": [True],
                "actual_yes_win_2": [True],
                "pair_win": [True],
                "market_pair_prob_independent": [0.25],
                "estimated_pair_prob_independent": [0.35],
                "pair_edge_independent": [0.10],
                "synthetic_independence_profit_per_dollar": [3.0],
                "side_combo": ["home_home"],
                "price_bucket_combo": ["favorite_favorite"],
            }
        )

        report, summary = summarize_parlay_pairs(pairs)

        self.assertFalse(report.empty)
        self.assertEqual(summary["status"], "blocked_too_few_pair_observations")
        self.assertFalse(summary["parlay_ready"])

    def test_apply_strategy_readiness_gate_blocks_when_no_strategy_ready(self) -> None:
        summary = {"status": "correlation_watchlist", "failed_checks": [], "parlay_ready": False}

        gated = apply_strategy_readiness_gate(summary, {"parlay_ready": 0})

        self.assertEqual(gated["status"], "blocked_strategy_readiness")
        self.assertIn("no_parlay_ready_individual_strategy", gated["failed_checks"])


if __name__ == "__main__":
    unittest.main()
