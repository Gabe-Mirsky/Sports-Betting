from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.interactive_dashboard import (  # noqa: E402
    available_seasons,
    available_teams,
    build_all_game_decisions,
    build_upcoming_display,
    build_upcoming_market_display,
    filter_backtest_trades,
    filter_predictions,
    format_money,
    format_pct,
    friendly_table,
    latest_game_date,
    latest_season_game_count,
    load_report_bundle,
    run_manual_market_backtest,
)


class TestInteractiveDashboard(unittest.TestCase):
    def test_available_filters_are_sorted(self) -> None:
        predictions = pd.DataFrame(
            {
                "season": [2024, 2023, 2024],
                "home_team_abbr": ["BOS", "NYK", "LAL"],
                "away_team_abbr": ["NYK", "BOS", "GSW"],
            }
        )
        trades = pd.DataFrame({"yes_team_abbr": ["GSW", "BOS"]})

        self.assertEqual(available_seasons(predictions), [2023, 2024])
        self.assertEqual(available_teams(predictions, trades), ["BOS", "GSW", "LAL", "NYK"])

    def test_filter_predictions_by_season_team_and_probability(self) -> None:
        predictions = pd.DataFrame(
            {
                "season": [2023, 2024, 2024],
                "home_team_abbr": ["BOS", "LAL", "DEN"],
                "away_team_abbr": ["NYK", "GSW", "BOS"],
                "model_home_win_prob": [0.70, 0.45, 0.62],
            }
        )

        filtered = filter_predictions(
            predictions,
            seasons=[2024],
            teams=["BOS"],
            min_home_prob=0.60,
            max_home_prob=0.80,
        )

        self.assertEqual(filtered["home_team_abbr"].tolist(), ["DEN"])

    def test_filter_backtest_trades_handles_string_booleans(self) -> None:
        trades = pd.DataFrame(
            {
                "home_team_abbr": ["BOS", "LAL", "DEN"],
                "away_team_abbr": ["NYK", "GSW", "BOS"],
                "yes_team_abbr": ["BOS", "GSW", "DEN"],
                "edge": [0.08, 0.02, 0.11],
                "trade": ["True", "False", "True"],
            }
        )

        filtered = filter_backtest_trades(trades, teams=["BOS"], min_edge=0.05, only_trades=True)

        self.assertEqual(filtered["yes_team_abbr"].tolist(), ["BOS", "DEN"])

    def test_format_helpers(self) -> None:
        self.assertEqual(format_money(106.375), "$106.38")
        self.assertEqual(format_pct(0.06375), "6.38%")

    def test_friendly_table_renames_and_formats_jargon_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "season": [2025],
                "home_team_abbr": ["BOS"],
                "away_team_abbr": ["NYK"],
                "model_yes_prob": [0.62],
                "market_prob": [0.55],
                "edge": [0.07],
                "trade": ["True"],
                "reason": ["edge_met"],
            }
        )

        display = friendly_table(raw)

        self.assertIn("Our Picked Team Win Chance", display.columns)
        self.assertEqual(display["Season"].iloc[0], "2025-26")
        self.assertEqual(display["Our Advantage"].iloc[0], "7.00%")
        self.assertEqual(display["Paper Pick?"].iloc[0], "Yes")
        self.assertEqual(display["Why"].iloc[0], "Advantage is big enough")

    def test_friendly_table_keeps_duplicate_meanings_unique(self) -> None:
        raw = pd.DataFrame(
            {
                "yes_mid_cents": [55],
                "price_cents": [55],
                "model_yes_prob": [0.62],
                "model_prob": [0.62],
            }
        )

        display = friendly_table(raw)

        self.assertEqual(len(display.columns), len(set(display.columns)))
        self.assertIn("Market Price (cents)", display.columns)
        self.assertIn("Price Used (cents)", display.columns)

    def test_build_all_game_decisions_shows_model_pick_and_market_status(self) -> None:
        predictions = pd.DataFrame(
            {
                "game_id": ["g1", "g2"],
                "game_date": ["2026-01-01", "2026-01-02"],
                "season": [2025, 2025],
                "home_team_abbr": ["BOS", "LAL"],
                "away_team_abbr": ["NYK", "GSW"],
                "model_home_win_prob": [0.70, 0.45],
                "model_away_win_prob": [0.30, 0.55],
                "actual_home_win": ["True", "False"],
            }
        )
        suggestions = pd.DataFrame(
            {
                "game_id": ["g1"],
                "yes_team_abbr": ["BOS"],
                "market_prob": [0.55],
                "edge": [0.15],
                "price_cents": [55],
                "trade": ["True"],
                "actual_yes_win": ["True"],
                "reason": ["edge_met"],
            }
        )

        decisions = build_all_game_decisions(predictions, suggestions)

        self.assertEqual(decisions["model_pick_team"].tolist(), ["BOS", "GSW"])
        self.assertEqual(decisions["model_pick_won"].tolist(), [True, True])
        self.assertEqual(decisions["paper_decision"].tolist(), ["Paper bet", "No market price loaded"])

    def test_build_upcoming_display_adds_pick_team(self) -> None:
        upcoming = pd.DataFrame(
            {
                "game_id": ["g1"],
                "game_date": ["2026-05-08"],
                "season": [2025],
                "home_team_abbr": ["BOS"],
                "away_team_abbr": ["NYK"],
                "model_home_win_prob": [0.41],
                "model_away_win_prob": [0.59],
            }
        )

        display = build_upcoming_display(upcoming)

        self.assertEqual(display["model_pick_team"].iloc[0], "NYK")
        self.assertEqual(display["upcoming_status"].iloc[0], "Scheduled")

    def test_build_upcoming_market_display_uses_model_pick_contract(self) -> None:
        upcoming = pd.DataFrame(
            {
                "game_id": ["g1"],
                "game_date": ["2026-05-08"],
                "season": [2025],
                "home_team_abbr": ["PHI"],
                "away_team_abbr": ["NYK"],
                "model_home_win_prob": [0.30],
                "model_away_win_prob": [0.70],
                "upcoming_status": ["7:00 pm ET"],
            }
        )
        suggestions = pd.DataFrame(
            {
                "game_id": ["g1", "g1"],
                "yes_team_abbr": ["PHI", "NYK"],
                "market_prob": [0.35, 0.65],
                "edge": [-0.05, 0.05],
                "price_cents": [35, 65],
                "trade": ["False", "True"],
                "reason": ["edge_below_threshold", "edge_met"],
            }
        )

        display = build_upcoming_market_display(upcoming, suggestions)

        self.assertEqual(display["model_pick_team"].iloc[0], "NYK")
        self.assertEqual(display["yes_team_abbr"].iloc[0], "NYK")
        self.assertAlmostEqual(display["price_cents"].iloc[0], 65)
        self.assertEqual(display["paper_decision"].iloc[0], "Paper bet")

    def test_latest_game_date_and_current_season_count(self) -> None:
        predictions = pd.DataFrame(
            {
                "game_date": ["2025-10-01", "2026-04-12", "2024-04-14"],
                "season": [2025, 2025, 2024],
            }
        )

        self.assertEqual(latest_game_date(predictions), "2026-04-12")
        self.assertEqual(latest_season_game_count(predictions), 2)

    def test_load_report_bundle_tolerates_missing_folder(self) -> None:
        bundle = load_report_bundle(PROJECT_ROOT / "does_not_exist")

        self.assertTrue(bundle.threshold_sweep.empty)
        self.assertEqual(bundle.model_metrics, {})

    def test_run_manual_market_backtest_matches_and_trades(self) -> None:
        predictions = pd.DataFrame(
            {
                "game_id": ["0022400001"],
                "game_date": ["2024-10-22"],
                "home_team_abbr": ["BOS"],
                "away_team_abbr": ["NYK"],
                "model_home_win_prob": [0.70],
                "model_away_win_prob": [0.30],
                "actual_home_win": [1],
            }
        )
        markets = pd.DataFrame(
            {
                "market_ticker": ["TEST-BOS"],
                "event_ticker": ["TEST"],
                "game_date": ["2024-10-22"],
                "home_team_abbr": ["BOS"],
                "away_team_abbr": ["NYK"],
                "yes_team_abbr": ["BOS"],
                "yes_mid_cents": [55],
                "settlement": ["yes"],
            }
        )

        result = run_manual_market_backtest(
            predictions,
            markets,
            starting_bankroll=100.0,
            edge_threshold=0.05,
            max_bet_fraction=0.03,
        )

        self.assertEqual(len(result["matched"]), 1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertTrue(bool(result["trades"]["trade"].iloc[0]))
        self.assertGreater(result["summary"]["ending_bankroll"], 100.0)


if __name__ == "__main__":
    unittest.main()
