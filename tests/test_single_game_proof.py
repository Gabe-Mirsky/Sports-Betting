from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.single_game_proof import build_single_game_proof_report  # noqa: E402


class TestSingleGameProof(unittest.TestCase):
    def test_single_game_proof_fails_when_clv_and_backtest_fail(self) -> None:
        gates, summary = build_single_game_proof_report(
            market_truth_summary={
                "matched_game_markets": 500,
                "usable_price_counts": {
                    "pregame_60m": 500,
                    "pregame_30m": 500,
                    "pregame_5m": 500,
                },
                "ticker_mapping_mismatch_count": 0,
                "wide_spread_count": 0,
                "low_liquidity_count": 0,
            },
            backtest_summary={"starting_bankroll": 100, "ending_bankroll": 90},
            clv_summary={"avg_clv_cents": -0.1, "positive_clv_rate": 0.40},
            readiness_summary={},
            readiness=pd.DataFrame([{"strategy": "raw_calibrated", "status": "watchlist", "months": 5}]),
            calibrated_trades=pd.DataFrame(
                [
                    {
                        "calibrated_trade": True,
                        "season": 2025,
                        "yes_team_abbr": "BOS",
                        "price_cents": 50,
                        "realized_profit_per_share": 0.10,
                    }
                ]
            ),
        )

        self.assertEqual(summary["status"], "not_proven")
        self.assertIn("strategy_backtest_profit", summary["failed_gates"])
        self.assertIn("average_clv", summary["failed_gates"])
        self.assertFalse(bool(summary["parlay_research_allowed"]))
        self.assertFalse(bool(gates[gates["gate"].eq("repeatability_months")]["passed"].iloc[0]))

    def test_single_game_proof_can_pass_hard_gates(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "calibrated_trade": True,
                    "season": 2024,
                    "yes_team_abbr": "BOS",
                    "price_cents": 45,
                    "realized_profit_per_share": 0.10,
                },
                {
                    "calibrated_trade": True,
                    "season": 2025,
                    "yes_team_abbr": "NYK",
                    "price_cents": 65,
                    "realized_profit_per_share": 0.10,
                },
            ]
        )

        _, summary = build_single_game_proof_report(
            market_truth_summary={
                "matched_game_markets": 500,
                "usable_price_counts": {
                    "pregame_60m": 500,
                    "pregame_30m": 500,
                    "pregame_5m": 500,
                },
                "ticker_mapping_mismatch_count": 0,
                "wide_spread_count": 0,
                "low_liquidity_count": 0,
            },
            backtest_summary={"starting_bankroll": 100, "ending_bankroll": 110},
            clv_summary={"avg_clv_cents": 0.2, "positive_clv_rate": 0.55},
            readiness_summary={},
            readiness=pd.DataFrame(
                [{"strategy": "raw_calibrated", "status": "paper_trade_candidate", "months": 6}]
            ),
            calibrated_trades=trades,
        )

        self.assertEqual(summary["status"], "single_game_edge_proven")

    def test_single_game_proof_can_target_clv_filtered_strategy(self) -> None:
        _, summary = build_single_game_proof_report(
            market_truth_summary={
                "matched_game_markets": 500,
                "usable_price_counts": {
                    "pregame_60m": 500,
                    "pregame_30m": 500,
                    "pregame_5m": 500,
                },
                "ticker_mapping_mismatch_count": 0,
                "wide_spread_count": 0,
                "low_liquidity_count": 0,
            },
            backtest_summary={"starting_bankroll": 100, "ending_bankroll": 120},
            clv_summary={"avg_clv_cents": 1.0, "positive_clv_rate": 0.55},
            readiness_summary={},
            readiness=pd.DataFrame(
                [{"strategy": "clv_filtered_calibrated", "status": "paper_trade_candidate", "months": 6}]
            ),
            calibrated_trades=pd.DataFrame(
                [
                    {
                        "clv_filtered_trade": True,
                        "calibrated_trade": True,
                        "season": 2024,
                        "yes_team_abbr": "BOS",
                        "price_cents": 45,
                        "realized_profit_per_share": 0.10,
                    },
                    {
                        "clv_filtered_trade": True,
                        "calibrated_trade": True,
                        "season": 2025,
                        "yes_team_abbr": "NYK",
                        "price_cents": 65,
                        "realized_profit_per_share": 0.10,
                    },
                ]
            ),
            strategy_name="clv_filtered_calibrated",
        )

        self.assertEqual(summary["strategy_under_test"], "clv_filtered_calibrated")

    def test_single_game_proof_can_target_defensive_strategy(self) -> None:
        _, summary = build_single_game_proof_report(
            market_truth_summary={
                "matched_game_markets": 500,
                "usable_price_counts": {
                    "pregame_60m": 500,
                    "pregame_30m": 500,
                    "pregame_5m": 500,
                },
                "ticker_mapping_mismatch_count": 0,
                "wide_spread_count": 0,
                "low_liquidity_count": 0,
            },
            backtest_summary={"starting_bankroll": 100, "ending_bankroll": 120},
            clv_summary={"avg_clv_cents": 1.0, "positive_clv_rate": 0.55},
            readiness_summary={},
            readiness=pd.DataFrame(
                [{"strategy": "defensive_clv_filtered", "status": "paper_trade_candidate", "months": 6}]
            ),
            calibrated_trades=pd.DataFrame(
                [
                    {
                        "defensive_trade": True,
                        "season": 2024,
                        "yes_team_abbr": "BOS",
                        "price_cents": 15,
                        "realized_profit_per_share": 0.10,
                    },
                    {
                        "defensive_trade": True,
                        "season": 2025,
                        "yes_team_abbr": "NYK",
                        "price_cents": 18,
                        "realized_profit_per_share": 0.10,
                    },
                ]
            ),
            strategy_name="defensive_clv_filtered",
        )

        self.assertEqual(summary["strategy_under_test"], "defensive_clv_filtered")


if __name__ == "__main__":
    unittest.main()
