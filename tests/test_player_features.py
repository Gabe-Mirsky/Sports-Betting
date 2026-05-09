from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from features.player_features import build_player_game_features  # noqa: E402
from features.team_features import build_modeling_dataset  # noqa: E402


class TestPlayerFeatures(unittest.TestCase):
    def _games(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "season": 2023,
                    "season_type": "Regular Season",
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 2,
                    "away_team_abbr": "BBB",
                    "home_points": 110,
                    "away_points": 100,
                    "home_win": 1,
                    "away_win": 0,
                },
                {
                    "game_id": "g2",
                    "game_date": "2024-01-03",
                    "season": 2023,
                    "season_type": "Regular Season",
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 3,
                    "away_team_abbr": "CCC",
                    "home_points": 90,
                    "away_points": 100,
                    "home_win": 0,
                    "away_win": 1,
                },
            ]
        )

    def _player_logs(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DATE": "2024-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "AAA",
                    "PLAYER_ID": "p1",
                    "PLAYER_NAME": "Alpha One",
                    "MIN": "30:00",
                    "PTS": 10,
                    "REB": 5,
                    "AST": 2,
                    "STL": 1,
                    "BLK": 0,
                    "TOV": 1,
                    "PLUS_MINUS": 8,
                },
                {
                    "GAME_ID": "g1",
                    "GAME_DATE": "2024-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "AAA",
                    "PLAYER_ID": "p2",
                    "PLAYER_NAME": "Alpha Two",
                    "MIN": "20:00",
                    "PTS": 8,
                    "REB": 3,
                    "AST": 4,
                    "STL": 0,
                    "BLK": 1,
                    "TOV": 2,
                    "PLUS_MINUS": 2,
                },
                {
                    "GAME_ID": "g1",
                    "GAME_DATE": "2024-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "AAA",
                    "PLAYER_ID": "p3",
                    "PLAYER_NAME": "Alpha Three",
                    "MIN": "10:00",
                    "PTS": 2,
                    "REB": 1,
                    "AST": 1,
                    "STL": 0,
                    "BLK": 0,
                    "TOV": 0,
                    "PLUS_MINUS": -1,
                },
                {
                    "GAME_ID": "c0",
                    "GAME_DATE": "2023-12-30",
                    "TEAM_ID": 3,
                    "TEAM_ABBREVIATION": "CCC",
                    "PLAYER_ID": "q1",
                    "PLAYER_NAME": "Charlie One",
                    "MIN": "24:00",
                    "PTS": 5,
                    "REB": 4,
                    "AST": 3,
                    "STL": 0,
                    "BLK": 0,
                    "TOV": 1,
                    "PLUS_MINUS": 0,
                },
                {
                    "GAME_ID": "g2",
                    "GAME_DATE": "2024-01-03",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "AAA",
                    "PLAYER_ID": "p1",
                    "PLAYER_NAME": "Alpha One",
                    "MIN": "1:00",
                    "PTS": 100,
                    "REB": 0,
                    "AST": 0,
                    "STL": 0,
                    "BLK": 0,
                    "TOV": 0,
                    "PLUS_MINUS": 50,
                },
            ]
        )

    def test_player_features_exclude_current_game_logs(self) -> None:
        features = build_player_game_features(self._games(), self._player_logs())
        row = features[features["game_id"].eq("g2")].iloc[0]

        self.assertAlmostEqual(row["home_player_top8_points_last10"], 20.0)
        self.assertNotAlmostEqual(row["home_player_top8_points_last10"], 120.0)
        self.assertAlmostEqual(row["away_player_top8_points_last10"], 5.0)
        self.assertAlmostEqual(row["player_top8_points_last10_diff"], 15.0)

    def test_modeling_dataset_includes_player_rotation_differences(self) -> None:
        modeling = build_modeling_dataset(self._games(), player_logs=self._player_logs())
        self.assertIn("player_top8_points_last10_diff", modeling.columns)
        row = modeling[modeling["game_id"].eq("g2")].iloc[0]
        self.assertAlmostEqual(row["player_top8_points_last10_diff"], 15.0)


if __name__ == "__main__":
    unittest.main()
