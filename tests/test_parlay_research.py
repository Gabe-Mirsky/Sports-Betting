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
from strategy.parlay_recommendations import build_parlay_recommendations  # noqa: E402
from strategy.parlay_recommendations import build_research_parlay_candidates  # noqa: E402


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

    def test_parlay_recommendations_exclude_same_game_and_rank_edges(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market": "BOS at NYK",
                    "recommendation": "Bet YES",
                    "side": "YES",
                    "yes_team": "NYK",
                    "home_team": "NYK",
                    "away_team": "BOS",
                    "calibrated_prob": 0.60,
                    "market_yes_ask": 52,
                    "market_no_ask": 49,
                    "final_edge": 0.08,
                    "market_ticker": "m1",
                },
                {
                    "game_id": "g2",
                    "market": "LAL at DEN",
                    "recommendation": "Bet NO",
                    "side": "NO",
                    "yes_team": "DEN",
                    "home_team": "DEN",
                    "away_team": "LAL",
                    "calibrated_prob": 0.40,
                    "market_yes_ask": 48,
                    "market_no_ask": 53,
                    "final_edge": 0.07,
                    "market_ticker": "m2",
                },
                {
                    "game_id": "g1",
                    "market": "BOS at NYK",
                    "recommendation": "Bet YES",
                    "side": "YES",
                    "yes_team": "BOS",
                    "home_team": "NYK",
                    "away_team": "BOS",
                    "calibrated_prob": 0.55,
                    "market_yes_ask": 50,
                    "market_no_ask": 51,
                    "final_edge": 0.05,
                    "market_ticker": "m3",
                },
            ]
        )

        recommendations, summary = build_parlay_recommendations(
            signals,
            proof_summary={"single_game_edge_proven": True},
            bankroll=100,
            min_combined_edge=0.01,
        )

        self.assertEqual(summary["status"], "ready_research_only")
        self.assertEqual(len(recommendations), 2)
        self.assertTrue((recommendations["leg_1_game"] != recommendations["leg_2_game"]).all())
        self.assertGreater(recommendations.loc[0, "combined_edge"], 0)
        self.assertGreater(recommendations.loc[0, "suggested_stake"], 0)

    def test_parlay_recommendations_block_when_single_game_not_proven(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "recommendation": "Bet YES",
                    "side": "YES",
                    "yes_team": "NYK",
                    "calibrated_prob": 0.60,
                    "market_yes_ask": 52,
                    "market_no_ask": 49,
                    "final_edge": 0.08,
                },
                {
                    "game_id": "g2",
                    "recommendation": "Bet YES",
                    "side": "YES",
                    "yes_team": "LAL",
                    "calibrated_prob": 0.58,
                    "market_yes_ask": 50,
                    "market_no_ask": 51,
                    "final_edge": 0.08,
                },
            ]
        )

        recommendations, summary = build_parlay_recommendations(
            signals,
            proof_summary={"single_game_edge_proven": False},
        )

        self.assertTrue(recommendations.empty)
        self.assertEqual(summary["status"], "blocked_single_game_edge_not_proven")
        self.assertFalse(summary["parlay_recommendations_allowed"])

    def test_research_parlays_generate_when_single_game_not_proven(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market": "BOS at NYK",
                    "market_ticker": "m1",
                    "recommendation_tier": "research_lean",
                    "research_side": "YES",
                    "research_price": 52,
                    "research_model_probability": 0.60,
                    "research_market_implied_probability": 0.52,
                    "edge": 0.08,
                    "yes_team": "NYK",
                    "home_team": "NYK",
                    "away_team": "BOS",
                },
                {
                    "game_id": "g2",
                    "market": "LAL at DEN",
                    "market_ticker": "m2",
                    "recommendation_tier": "paper_trade_candidate",
                    "research_side": "NO",
                    "research_price": 48,
                    "research_model_probability": 0.62,
                    "research_market_implied_probability": 0.48,
                    "edge": 0.14,
                    "yes_team": "DEN",
                    "home_team": "DEN",
                    "away_team": "LAL",
                },
            ]
        )

        approved, approved_summary = build_parlay_recommendations(
            signals.assign(recommendation="Research Lean YES", side=""),
            proof_summary={"single_game_edge_proven": False},
        )
        research, research_summary = build_research_parlay_candidates(signals, parlay_tier="research_parlay")

        self.assertTrue(approved.empty)
        self.assertEqual(approved_summary["status"], "blocked_single_game_edge_not_proven")
        self.assertEqual(len(research), 1)
        self.assertTrue(bool(research.loc[0, "research_only"]))
        self.assertFalse(bool(research.loc[0, "approved"]))
        self.assertEqual(research.loc[0, "parlay_tier"], "research_parlay")
        self.assertNotIn("suggested_stake", research.columns)
        self.assertEqual(research_summary["status"], "research_only_generated")

    def test_paper_parlays_use_only_paper_trade_candidate_legs(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market": "BOS at NYK",
                    "market_ticker": "m1",
                    "recommendation_tier": "research_lean",
                    "research_side": "YES",
                    "research_price": 52,
                    "research_model_probability": 0.60,
                    "research_market_implied_probability": 0.52,
                    "edge": 0.08,
                    "yes_team": "NYK",
                },
                {
                    "game_id": "g2",
                    "market": "LAL at DEN",
                    "market_ticker": "m2",
                    "recommendation_tier": "paper_trade_candidate",
                    "research_side": "YES",
                    "research_price": 50,
                    "research_model_probability": 0.62,
                    "research_market_implied_probability": 0.50,
                    "edge": 0.12,
                    "yes_team": "LAL",
                },
                {
                    "game_id": "g3",
                    "market": "MIA at ORL",
                    "market_ticker": "m3",
                    "recommendation_tier": "paper_trade_candidate",
                    "research_side": "NO",
                    "research_price": 45,
                    "research_model_probability": 0.58,
                    "research_market_implied_probability": 0.45,
                    "edge": 0.13,
                    "yes_team": "ORL",
                },
            ]
        )

        paper, summary = build_research_parlay_candidates(signals, parlay_tier="paper_parlay")

        self.assertEqual(len(paper), 1)
        self.assertEqual(summary["source_recommendation_tiers"], ["paper_trade_candidate"])
        self.assertEqual(paper.loc[0, "parlay_tier"], "paper_parlay")
        self.assertIn("LAL at DEN", paper.loc[0, "legs"])
        self.assertIn("MIA at ORL", paper.loc[0, "legs"])
        self.assertNotIn("BOS at NYK", paper.loc[0, "legs"])


if __name__ == "__main__":
    unittest.main()
