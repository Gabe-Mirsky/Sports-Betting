from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from features.rolling_stats import add_team_rolling_features, games_to_team_game_long  # noqa: E402
from features.team_features import build_upcoming_modeling_dataset  # noqa: E402


class TestFeaturesNoLeakage(unittest.TestCase):
    def test_rolling_margin_excludes_current_game(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": f"g{i}",
                    "game_date": f"2024-01-0{i}",
                    "season": 2023,
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 100 + i,
                    "away_team_abbr": f"B{i}",
                    "home_points": 100 + margin,
                    "away_points": 100,
                    "home_win": 1,
                    "away_win": 0,
                }
                for i, margin in enumerate([10, 20, 30, 40], start=1)
            ]
        )
        team_games = games_to_team_game_long(games)
        featured = add_team_rolling_features(team_games, windows=[3])
        row = featured[(featured["team_id"] == 1) & (featured["game_id"] == "g4")].iloc[0]

        self.assertAlmostEqual(row["last_3_point_diff"], 20.0)
        self.assertNotAlmostEqual(row["last_3_point_diff"], 30.0)

    def test_first_game_has_missing_rolling_feature(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "season": 2023,
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 2,
                    "away_team_abbr": "BBB",
                    "home_points": 110,
                    "away_points": 100,
                    "home_win": 1,
                    "away_win": 0,
                }
            ]
        )
        featured = add_team_rolling_features(games_to_team_game_long(games), windows=[3])
        self.assertTrue(pd.isna(featured.loc[featured["team_id"] == 1, "last_3_point_diff"].iloc[0]))

    def test_box_score_rolling_features_exclude_current_game(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": f"g{i}",
                    "game_date": f"2024-01-0{i}",
                    "season": 2023,
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 100 + i,
                    "away_team_abbr": f"B{i}",
                    "home_points": 100 + i,
                    "away_points": 100,
                    "home_win": 1,
                    "away_win": 0,
                    "home_fg_pct": value,
                    "away_fg_pct": 0.40,
                }
                for i, value in enumerate([0.40, 0.50, 0.60], start=1)
            ]
        )

        featured = add_team_rolling_features(games_to_team_game_long(games), windows=[3])
        row = featured[(featured["team_id"] == 1) & (featured["game_id"] == "g3")].iloc[0]

        self.assertAlmostEqual(row["last_5_fg_pct"], 0.45)
        self.assertNotAlmostEqual(row["last_5_fg_pct"], 0.50)

    def test_upcoming_features_use_completed_history_only(self) -> None:
        completed = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-01-01",
                    "season": 2025,
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
                    "game_date": "2026-01-03",
                    "season": 2025,
                    "season_type": "Regular Season",
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 3,
                    "away_team_abbr": "CCC",
                    "home_points": 120,
                    "away_points": 100,
                    "home_win": 1,
                    "away_win": 0,
                },
            ]
        )
        upcoming = pd.DataFrame(
            [
                {
                    "game_id": "g3",
                    "game_date": "2026-01-05",
                    "season": 2025,
                    "season_type": "Regular Season",
                    "home_team_id": 1,
                    "home_team_abbr": "AAA",
                    "away_team_id": 4,
                    "away_team_abbr": "DDD",
                }
            ]
        )

        features = build_upcoming_modeling_dataset(completed, upcoming)

        self.assertAlmostEqual(features["home_last_3_point_diff"].iloc[0], 15.0)
        self.assertEqual(features["home_rest_days"].iloc[0], 2)
        self.assertIn("elo_home_win_prob", features.columns)


if __name__ == "__main__":
    unittest.main()
