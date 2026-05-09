from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.injury_availability import (  # noqa: E402
    build_game_availability_features,
    enrich_availability_reports_with_player_impact,
)
from features.team_features import build_modeling_dataset  # noqa: E402


class TestInjuryAvailability(unittest.TestCase):
    def _games(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": pd.Timestamp("2026-01-10"),
                    "season": 2025,
                    "season_type": "Regular Season",
                    "home_team_id": 1,
                    "home_team_abbr": "BOS",
                    "away_team_id": 2,
                    "away_team_abbr": "NYK",
                    "home_points": 110,
                    "away_points": 100,
                    "home_win": True,
                },
                {
                    "game_id": "2",
                    "game_date": pd.Timestamp("2026-01-12"),
                    "season": 2025,
                    "season_type": "Regular Season",
                    "home_team_id": 2,
                    "home_team_abbr": "NYK",
                    "away_team_id": 1,
                    "away_team_abbr": "BOS",
                    "home_points": 105,
                    "away_points": 101,
                    "home_win": True,
                },
            ]
        )

    def test_build_game_availability_features_uses_only_reports_by_game_date(self) -> None:
        reports = pd.DataFrame(
            [
                {
                    "report_date": "2026-01-10",
                    "game_date": "2026-01-10",
                    "team_abbr": "BOS",
                    "player_name": "Player One",
                    "status": "Out",
                    "impact_weight": 32,
                },
                {
                    "report_date": "2026-01-11",
                    "game_date": "2026-01-10",
                    "team_abbr": "BOS",
                    "player_name": "Player Two",
                    "status": "Out",
                    "impact_weight": 10,
                },
                {
                    "report_date": "2026-01-10",
                    "game_date": "2026-01-10",
                    "team_abbr": "NYK",
                    "player_name": "Player Three",
                    "status": "Questionable",
                    "impact_weight": 24,
                },
            ]
        )

        features = build_game_availability_features(self._games(), reports)
        row = features[features["game_id"].eq("1")].iloc[0]

        self.assertEqual(row["home_availability_players_out"], 1)
        self.assertEqual(row["away_availability_players_questionable"], 1)
        self.assertEqual(row["availability_questionable_or_worse_diff"], 0)
        self.assertEqual(row["home_availability_out_weighted"], 32)
        self.assertEqual(row["away_availability_questionable_weighted"], 24)
        self.assertEqual(row["availability_projected_minutes_lost_diff"], 20)

    def test_modeling_dataset_includes_availability_differences_when_reports_exist(self) -> None:
        reports = pd.DataFrame(
            [
                {
                    "report_date": "2026-01-10",
                    "game_date": "2026-01-10",
                    "team_abbr": "BOS",
                    "player_name": "Player One",
                    "status": "Out",
                }
            ]
        )

        modeling = build_modeling_dataset(self._games(), availability_reports=reports)

        self.assertIn("availability_players_out_diff", modeling.columns)
        self.assertEqual(modeling.loc[0, "availability_players_out_diff"], 1)
        self.assertIn("availability_projected_minutes_lost_diff", modeling.columns)
        self.assertEqual(modeling.loc[0, "availability_projected_minutes_lost_diff"], 1)

    def test_availability_impact_weight_can_use_expected_minutes_column(self) -> None:
        reports = pd.DataFrame(
            [
                {
                    "report_date": "2026-01-10",
                    "game_date": "2026-01-10",
                    "team_abbr": "BOS",
                    "player_name": "Starter",
                    "status": "Doubtful",
                    "expected_minutes": 28,
                }
            ]
        )

        features = build_game_availability_features(self._games(), reports)
        row = features[features["game_id"].eq("1")].iloc[0]

        self.assertEqual(row["home_availability_doubtful_weighted"], 28)
        self.assertEqual(row["home_availability_projected_minutes_lost"], 21)

    def test_availability_impact_can_be_enriched_from_prior_player_logs(self) -> None:
        reports = pd.DataFrame(
            [
                {
                    "report_date": "2026-01-10",
                    "game_date": "2026-01-10",
                    "team_abbr": "BOS",
                    "player_name": "Starter One",
                    "status": "Out",
                }
            ]
        )
        player_logs = pd.DataFrame(
            [
                {
                    "GAME_ID": "prev1",
                    "GAME_DATE": "2026-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "BOS",
                    "PLAYER_ID": 101,
                    "PLAYER_NAME": "Starter One",
                    "MIN": 30,
                    "PTS": 20,
                    "REB": 5,
                    "AST": 5,
                    "STL": 1,
                    "BLK": 0,
                    "TOV": 2,
                    "PLUS_MINUS": 4,
                    "SEASON_ID": 22025,
                },
                {
                    "GAME_ID": "prev2",
                    "GAME_DATE": "2026-01-03",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "BOS",
                    "PLAYER_ID": 101,
                    "PLAYER_NAME": "Starter One",
                    "MIN": 34,
                    "PTS": 22,
                    "REB": 4,
                    "AST": 6,
                    "STL": 0,
                    "BLK": 1,
                    "TOV": 1,
                    "PLUS_MINUS": 8,
                    "SEASON_ID": 22025,
                },
                {
                    "GAME_ID": "future",
                    "GAME_DATE": "2026-01-10",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "BOS",
                    "PLAYER_ID": 101,
                    "PLAYER_NAME": "Starter One",
                    "MIN": 5,
                    "PTS": 0,
                    "REB": 0,
                    "AST": 0,
                    "STL": 0,
                    "BLK": 0,
                    "TOV": 0,
                    "PLUS_MINUS": 0,
                    "SEASON_ID": 22025,
                },
            ]
        )

        enriched = enrich_availability_reports_with_player_impact(reports, player_logs)
        features = build_game_availability_features(self._games(), enriched)
        row = features[features["game_id"].eq("1")].iloc[0]

        self.assertEqual(enriched.loc[0, "impact_prior_games"], 2)
        self.assertAlmostEqual(enriched.loc[0, "impact_weight"], 32.0)
        self.assertEqual(row["home_availability_out_weighted"], 32.0)


if __name__ == "__main__":
    unittest.main()
