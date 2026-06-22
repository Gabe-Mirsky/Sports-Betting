from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.probability_shrinkage import adjusted_edge, shrink_probability
from strategy.shrinkage_policy_sweep import (
    ShrinkagePolicy,
    build_candidate_rows,
    run_side_specific_shrinkage_sweep,
    select_policy_trades,
)
from strategy.uncertainty_penalty import apply_prior_penalties, calculate_prior_penalty


def _market_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_date": "2026-01-01",
                "game_id": "1",
                "market_ticker": "A",
                "home_team_abbr": "AAA",
                "away_team_abbr": "BBB",
                "yes_team_abbr": "AAA",
                "model_yes_prob": 0.72,
                "yes_ask": 60,
                "yes_bid": 58,
                "actual_yes_win": True,
                "volume": 200,
                "clv_reference_price_cents": 63,
                "clv_reference_no_price_cents": 39,
            },
            {
                "game_date": "2026-02-01",
                "game_id": "2",
                "market_ticker": "B",
                "home_team_abbr": "CCC",
                "away_team_abbr": "DDD",
                "yes_team_abbr": "CCC",
                "model_yes_prob": 0.70,
                "yes_ask": 60,
                "yes_bid": 58,
                "actual_yes_win": False,
                "volume": 200,
                "clv_reference_price_cents": 55,
                "clv_reference_no_price_cents": 45,
            },
            {
                "game_date": "2026-03-01",
                "game_id": "3",
                "market_ticker": "C",
                "home_team_abbr": "EEE",
                "away_team_abbr": "FFF",
                "yes_team_abbr": "EEE",
                "model_yes_prob": 0.30,
                "yes_ask": 42,
                "yes_bid": 40,
                "actual_yes_win": False,
                "volume": 80,
                "clv_reference_price_cents": 39,
                "clv_reference_no_price_cents": 63,
            },
        ]
    )


class TestSideSpecificShrinkage(unittest.TestCase):
    def test_shrinkage_formula_and_clamp(self) -> None:
        self.assertAlmostEqual(shrink_probability(0.80, 0.50, 0.50), 0.65)
        self.assertAlmostEqual(shrink_probability(1.50, 0.80, 1.00), 1.0)
        self.assertAlmostEqual(shrink_probability(-0.50, 0.20, 1.00), 0.0)

    def test_yes_and_no_shrink_factors_apply_separately(self) -> None:
        candidates = build_candidate_rows(_market_rows())
        policy = ShrinkagePolicy(
            yes_shrink_factor=0.50,
            no_shrink_factor=0.25,
            min_edge=0.01,
            uncertainty_penalty_mode="none",
            min_prior_samples=20,
        )
        selected = select_policy_trades(candidates, policy)

        yes = selected[selected["side"].eq("YES")].iloc[0]
        no = selected[selected["side"].eq("NO")].iloc[0]

        self.assertAlmostEqual(yes["shrink_factor"], 0.50)
        self.assertAlmostEqual(no["shrink_factor"], 0.25)

    def test_edge_recalculation_after_shrinkage_and_penalty(self) -> None:
        self.assertAlmostEqual(adjusted_edge(0.80, 0.50, 0.50, uncertainty_penalty=0.02), 0.13)

    def test_uncertainty_penalty_reduces_edge(self) -> None:
        prior = pd.DataFrame(
            {
                "clv_cents": [-2, -1, 1, -3],
                "adjusted_probability": [0.60, 0.60, 0.60, 0.60],
                "contract_won": [0, 0, 1, 0],
            }
        )
        penalty = calculate_prior_penalty(prior, min_prior_samples=2, conservative_default=0.03)

        self.assertGreater(penalty, 0.0)
        self.assertLess(adjusted_edge(0.80, 0.50, 1.0, penalty), adjusted_edge(0.80, 0.50, 1.0, 0.0))

    def test_small_sample_conservative_fallback(self) -> None:
        current = pd.DataFrame({"side": ["YES"], "price_bucket": ["40-55"]})
        prior = pd.DataFrame({"side": ["YES"], "price_bucket": ["40-55"], "clv_cents": [10]})

        penalized = apply_prior_penalties(
            current,
            prior,
            mode="side+price_bucket",
            min_prior_samples=20,
            conservative_default=0.03,
        )

        self.assertAlmostEqual(float(penalized.loc[0, "uncertainty_penalty"]), 0.03)
        self.assertEqual(penalized.loc[0, "penalty_source"], "conservative_default_small_sample")

    def test_prior_period_only_penalty_blocks_future_leakage(self) -> None:
        markets = _market_rows()
        candidates = build_candidate_rows(markets)
        policy = ShrinkagePolicy(
            yes_shrink_factor=1.0,
            no_shrink_factor=1.0,
            min_edge=0.03,
            uncertainty_penalty_mode="side-only",
            min_prior_samples=2,
        )

        selected = select_policy_trades(candidates, policy)
        february = selected[selected["game_date"].dt.to_period("M").astype(str).eq("2026-02")]

        self.assertTrue((february["penalty_source"] != "prior_bucket").all())

    def test_sweep_keeps_proof_and_parlays_blocked_when_not_proven(self) -> None:
        policies = [
            ShrinkagePolicy(
                yes_shrink_factor=0.5,
                no_shrink_factor=0.5,
                min_edge=0.03,
                uncertainty_penalty_mode="none",
                min_prior_samples=20,
            )
        ]
        sweep, _, summary, _ = run_side_specific_shrinkage_sweep(
            _market_rows(),
            policies=policies,
            baseline={"profit": -10, "average_clv_cents": -1, "positive_clv_rate": 0.2, "yes_profit": -10},
        )

        self.assertFalse(summary["single_game_edge_proven"])
        self.assertEqual(summary["parlay_status"], "blocked_single_game_edge_not_proven")
        self.assertFalse(summary["parlay_recommendations_allowed"])
        self.assertEqual(int(sweep.iloc[0]["fair_price_actionable_bets_after_proof_gate"]), 0)


if __name__ == "__main__":
    unittest.main()
