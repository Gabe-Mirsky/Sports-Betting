from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.forward import build_forward_recommendations  # noqa: E402


class TestForwardRecommendations(unittest.TestCase):
    def test_forward_recommendations_size_paper_bets_when_gate_allows(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-05-08 19:00:00",
                    "season_type": "Playoffs",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "model_home_win_prob": 0.70,
                    "model_away_win_prob": 0.30,
                    "upcoming_status": "7:00 pm ET",
                }
            ]
        )
        suggestions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "M1",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.70,
                    "market_prob": 0.55,
                    "edge": 0.15,
                    "price_cents": 55,
                    "trade": True,
                    "reason": "edge_met",
                }
            ]
        )

        rows, summary = build_forward_recommendations(
            predictions,
            suggestions,
            readiness_summary={"paper_trade_candidates": 1},
            starting_bankroll=100,
            max_bet_fraction=0.03,
            as_of_date="2026-05-08",
        )

        self.assertEqual(summary["paper_bets"], 1)
        self.assertEqual(int(rows.loc[0, "paper_shares"]), 5)
        self.assertAlmostEqual(float(rows.loc[0, "paper_amount_risked"]), 2.75)
        self.assertEqual(rows.loc[0, "recommendation"], "Paper bet")

    def test_forward_recommendations_show_watchlist_when_gate_blocks(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-05-08",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "model_home_win_prob": 0.70,
                    "model_away_win_prob": 0.30,
                }
            ]
        )
        suggestions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "M1",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.70,
                    "market_prob": 0.55,
                    "edge": 0.15,
                    "price_cents": 55,
                    "trade": True,
                    "reason": "edge_met",
                }
            ]
        )

        rows, summary = build_forward_recommendations(
            predictions,
            suggestions,
            readiness_summary={"paper_trade_candidates": 0},
            starting_bankroll=100,
            max_bet_fraction=0.03,
            as_of_date="2026-05-08",
        )

        self.assertEqual(summary["paper_bets"], 0)
        self.assertEqual(summary["hypothetical_paper_bets"], 1)
        self.assertEqual(float(rows.loc[0, "paper_amount_risked"]), 0.0)
        self.assertEqual(rows.loc[0, "recommendation"], "Watchlist only - readiness gate")

    def test_forward_recommendations_block_in_sample_sweep_without_walk_forward_validation(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-05-08",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "model_home_win_prob": 0.70,
                    "model_away_win_prob": 0.30,
                }
            ]
        )
        suggestions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "M1",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.70,
                    "market_prob": 0.55,
                    "edge": 0.15,
                    "price_cents": 55,
                    "trade": True,
                    "reason": "edge_met",
                }
            ]
        )

        rows, summary = build_forward_recommendations(
            predictions,
            suggestions,
            readiness_summary={"paper_trade_candidates": 0},
            rule_sweep_summary={
                "best_rule": "edge>=0.02, roi>=0.00, history>=0, price=10-90c",
                "best_rule_status": "watchlist",
                "best_rule_params": {
                    "min_edge": 0.02,
                    "min_expected_roi": 0.0,
                    "min_edge_bin_history_rows": 0,
                    "min_price_cents": 10,
                    "max_price_cents": 90,
                },
            },
            starting_bankroll=100,
            max_bet_fraction=0.03,
            as_of_date="2026-05-08",
        )

        self.assertFalse(bool(rows.loc[0, "passes_best_sweep_rule"]))
        self.assertEqual(summary["best_sweep_rule_passes"], 0)
        self.assertEqual(summary["rule_validation_status"], "missing")
        self.assertEqual(rows.loc[0, "recommendation"], "Watchlist only - readiness gate")

    def test_forward_recommendations_surface_rule_passes_after_walk_forward_validation(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-05-08",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "model_home_win_prob": 0.70,
                    "model_away_win_prob": 0.30,
                }
            ]
        )
        suggestions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "M1",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.70,
                    "market_prob": 0.55,
                    "edge": 0.15,
                    "price_cents": 55,
                    "trade": True,
                    "reason": "edge_met",
                }
            ]
        )

        rows, summary = build_forward_recommendations(
            predictions,
            suggestions,
            readiness_summary={"paper_trade_candidates": 0},
            rule_sweep_summary={
                "best_rule": "edge>=0.02, roi>=0.00, history>=0, price=10-90c",
                "best_rule_status": "watchlist",
                "best_rule_params": {
                    "min_edge": 0.02,
                    "min_expected_roi": 0.0,
                    "min_edge_bin_history_rows": 0,
                    "min_price_cents": 10,
                    "max_price_cents": 90,
                },
            },
            rule_validation_summary={"status": "walk_forward_candidate", "signals": 150, "positive_months": 5, "months": 6},
            starting_bankroll=100,
            max_bet_fraction=0.03,
            as_of_date="2026-05-08",
        )

        self.assertTrue(bool(rows.loc[0, "passes_best_sweep_rule"]))
        self.assertEqual(summary["best_sweep_rule_passes"], 1)
        self.assertEqual(summary["rule_validation_status"], "walk_forward_candidate")
        self.assertEqual(rows.loc[0, "recommendation"], "Watchlist only - passes best sweep rule")


if __name__ == "__main__":
    unittest.main()
