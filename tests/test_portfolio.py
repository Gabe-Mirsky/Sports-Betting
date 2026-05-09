from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.portfolio import optimize_individual_bet_slate, prepare_portfolio_candidates  # noqa: E402


class TestPortfolio(unittest.TestCase):
    def test_prepare_candidates_keeps_edge_met_trade_rows(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "trade": True,
                    "price_cents": 50,
                    "model_yes_prob": 0.60,
                    "edge": 0.10,
                    "actual_yes_win": True,
                },
                {
                    "date": "2025-01-01",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "trade": False,
                    "price_cents": 50,
                    "model_yes_prob": 0.60,
                    "edge": 0.10,
                    "actual_yes_win": True,
                },
            ]
        )

        candidates = prepare_portfolio_candidates(trades, min_edge=0.05)

        self.assertEqual(candidates["market_ticker"].tolist(), ["A"])
        self.assertGreater(float(candidates.loc[0, "expected_roi"]), 0)

    def test_prepare_candidates_can_use_calibrated_trade_column(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "trade": False,
                    "calibrated_trade": True,
                    "price_cents": 50,
                    "model_yes_prob": 0.60,
                    "edge": 0.10,
                    "calibrated_expected_roi": 0.08,
                    "actual_yes_win": True,
                }
            ]
        )

        candidates = prepare_portfolio_candidates(
            trades,
            min_edge=0.05,
            trade_column="calibrated_trade",
            expected_roi_column="calibrated_expected_roi",
        )

        self.assertEqual(candidates["market_ticker"].tolist(), ["A"])
        self.assertAlmostEqual(float(candidates.loc[0, "selection_expected_roi"]), 0.08)

    def test_optimizer_limits_one_market_per_game_and_reports_timeline(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "trade": True,
                    "price_cents": 50,
                    "market_prob": 0.50,
                    "model_yes_prob": 0.70,
                    "edge": 0.20,
                    "actual_yes_win": True,
                },
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "B",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "BBB",
                    "trade": True,
                    "price_cents": 40,
                    "market_prob": 0.40,
                    "model_yes_prob": 0.60,
                    "edge": 0.20,
                    "actual_yes_win": False,
                },
                {
                    "date": "2025-01-02",
                    "game_id": "g2",
                    "market_ticker": "C",
                    "home_team_abbr": "CCC",
                    "away_team_abbr": "DDD",
                    "yes_team_abbr": "CCC",
                    "trade": True,
                    "price_cents": 50,
                    "market_prob": 0.50,
                    "model_yes_prob": 0.60,
                    "edge": 0.10,
                    "actual_yes_win": False,
                },
            ]
        )

        selected, slates, summary = optimize_individual_bet_slate(
            trades,
            starting_bankroll=100,
            max_trades_per_slate=2,
            max_markets_per_game=1,
            max_markets_per_team=2,
            max_bet_fraction=0.03,
            max_slate_fraction=0.10,
        )

        self.assertEqual(summary["trade_timeline"], "2025-01-01 to 2025-01-02")
        self.assertEqual(int(selected[selected["game_id"].eq("g1")].shape[0]), 1)
        self.assertEqual(summary["num_selected_trades"], 2)
        self.assertEqual(int(slates["selected_trades"].sum()), 2)

    def test_optimizer_limits_team_exposure_on_same_slate(self) -> None:
        trades = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "trade": True,
                    "price_cents": 50,
                    "market_prob": 0.50,
                    "model_yes_prob": 0.70,
                    "edge": 0.20,
                    "actual_yes_win": True,
                },
                {
                    "date": "2025-01-01",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "CCC",
                    "yes_team_abbr": "AAA",
                    "trade": True,
                    "price_cents": 50,
                    "market_prob": 0.50,
                    "model_yes_prob": 0.69,
                    "edge": 0.19,
                    "actual_yes_win": True,
                },
            ]
        )

        selected, slates, summary = optimize_individual_bet_slate(
            trades,
            starting_bankroll=100,
            max_trades_per_slate=5,
            max_markets_per_game=1,
            max_markets_per_team=1,
            max_bet_fraction=0.03,
            max_slate_fraction=0.20,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(int(slates["rejected_by_team_cap"].sum()), 1)
        self.assertEqual(summary["rejected_by_team_cap"], 1)


if __name__ == "__main__":
    unittest.main()
