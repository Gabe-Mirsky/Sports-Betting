from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.fair_price import (  # noqa: E402
    apply_single_game_proof_gate,
    build_fair_price_signals,
    summarize_fair_price_signals,
)


class TestFairPrice(unittest.TestCase):
    def test_build_fair_price_signals_can_recommend_yes_or_no(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2026-01-01",
                    "game_id": "g1",
                    "market_ticker": "YES_EDGE",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.67,
                    "yes_bid": 52,
                    "yes_ask": 54,
                    "volume": 100,
                },
                {
                    "game_date": "2026-01-02",
                    "game_id": "g2",
                    "market_ticker": "NO_EDGE",
                    "home_team_abbr": "LAL",
                    "away_team_abbr": "DEN",
                    "yes_team_abbr": "LAL",
                    "model_yes_prob": 0.25,
                    "yes_bid": 63,
                    "yes_ask": 65,
                    "volume": 100,
                },
            ]
        )

        signals = build_fair_price_signals(
            markets,
            edge_threshold=0.03,
            fee_penalty=0.0,
            uncertainty_penalty=0.0,
            spread_penalty_fraction=0.0,
        )
        summary = summarize_fair_price_signals(signals)

        self.assertEqual(summary["bets"], 2)
        self.assertEqual(summary["yes_bets"], 1)
        self.assertEqual(summary["no_bets"], 1)
        self.assertEqual(set(signals["side"]), {"YES", "NO"})
        self.assertEqual(set(signals["recommendation_tier"]), {"paper_trade_candidate"})
        self.assertIn("research_side", signals.columns)
        self.assertIn("market_implied_probability", signals.columns)
        self.assertIn("edge", signals.columns)
        no_row = signals.loc[signals["market_ticker"].eq("NO_EDGE")].iloc[0]
        self.assertAlmostEqual(float(no_row["research_model_probability"]), 0.75)
        self.assertAlmostEqual(float(no_row["research_market_implied_probability"]), 0.37)

    def test_fair_price_blocks_wide_spread(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "WIDE",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.80,
                    "yes_bid": 40,
                    "yes_ask": 70,
                    "volume": 100,
                }
            ]
        )

        signals = build_fair_price_signals(markets, max_spread_cents=10)

        self.assertEqual(signals.loc[0, "recommendation"], "No bet")
        self.assertEqual(signals.loc[0, "main_reason"], "bid_ask_spread_too_wide")
        self.assertEqual(signals.loc[0, "recommendation_tier"], "none")

    def test_proof_gate_blocks_action_labels_but_keeps_ungated_audit(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "YES_EDGE",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.67,
                    "yes_bid": 52,
                    "yes_ask": 54,
                    "volume": 100,
                }
            ]
        )

        raw = build_fair_price_signals(
            markets,
            edge_threshold=0.03,
            fee_penalty=0.0,
            uncertainty_penalty=0.0,
            spread_penalty_fraction=0.0,
        )
        gated = apply_single_game_proof_gate(raw, single_game_edge_proven=False, proof_status="not_proven")
        summary = summarize_fair_price_signals(gated)

        self.assertEqual(gated.loc[0, "recommendation"], "Paper Trade Candidate YES")
        self.assertEqual(gated.loc[0, "main_reason"], "edge_survives_price_fee_spread_uncertainty_screens")
        self.assertEqual(gated.loc[0, "main_risk"], "single_game_edge_not_proven")
        self.assertEqual(gated.loc[0, "blocked_reason"], "single_game_edge_not_proven")
        self.assertEqual(gated.loc[0, "side"], "")
        self.assertTrue(pd.isna(gated.loc[0, "price"]))
        self.assertEqual(gated.loc[0, "ungated_side"], "YES")
        self.assertEqual(gated.loc[0, "research_side"], "YES")
        self.assertEqual(gated.loc[0, "recommendation_tier"], "paper_trade_candidate")
        self.assertEqual(summary["bets"], 0)
        self.assertEqual(summary["ungated_bets"], 1)
        self.assertEqual(summary["approved_bets_count"], 0)
        self.assertEqual(summary["paper_trade_candidates_count"], 1)
        self.assertEqual(summary["research_leans_count"], 0)
        self.assertEqual(summary["blocked_reason"], "single_game_edge_not_proven")
        self.assertEqual(summary["proof_status"], "not_proven")
        self.assertEqual(summary["proof_gate_status"], "not_proven")

    def test_research_lean_allowed_when_proof_gate_fails(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "LEAN",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.58,
                    "yes_bid": 52,
                    "yes_ask": 54,
                    "volume": 100,
                }
            ]
        )

        raw = build_fair_price_signals(
            markets,
            edge_threshold=0.03,
            paper_trade_min_edge=0.08,
            fee_penalty=0.0,
            uncertainty_penalty=0.0,
            spread_penalty_fraction=0.0,
        )
        gated = apply_single_game_proof_gate(raw, single_game_edge_proven=False, proof_status="not_proven")

        self.assertEqual(gated.loc[0, "recommendation_tier"], "research_lean")
        self.assertEqual(gated.loc[0, "recommendation"], "Research Lean YES")
        self.assertEqual(gated.loc[0, "side"], "")
        self.assertEqual(float(gated.loc[0, "research_price"]), 54.0)
        self.assertAlmostEqual(float(gated.loc[0, "market_implied_probability"]), 0.54)
        self.assertAlmostEqual(float(gated.loc[0, "edge"]), float(gated.loc[0, "final_edge"]))

    def test_proven_gate_promotes_only_to_approved_bet_tier(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "YES_EDGE",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.67,
                    "yes_bid": 52,
                    "yes_ask": 54,
                    "volume": 100,
                }
            ]
        )

        raw = build_fair_price_signals(
            markets,
            edge_threshold=0.03,
            fee_penalty=0.0,
            uncertainty_penalty=0.0,
            spread_penalty_fraction=0.0,
        )
        approved = apply_single_game_proof_gate(raw, single_game_edge_proven=True, proof_status="proven")
        summary = summarize_fair_price_signals(approved)

        self.assertEqual(approved.loc[0, "recommendation_tier"], "approved_bet")
        self.assertEqual(approved.loc[0, "recommendation"], "Bet YES")
        self.assertEqual(summary["bets"], 1)
        self.assertEqual(summary["approved_bets_count"], 1)
        self.assertEqual(summary["paper_trade_candidates_count"], 0)


if __name__ == "__main__":
    unittest.main()
