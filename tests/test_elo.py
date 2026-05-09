from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from features.elo import (  # noqa: E402
    add_elo_features,
    add_upcoming_elo_features,
    current_elo_state,
    expected_home_win_prob,
    update_elo_ratings,
)


class TestElo(unittest.TestCase):
    def test_favorite_has_probability_above_half(self) -> None:
        probability = expected_home_win_prob(1600, 1500, home_court_advantage=60)
        self.assertGreater(probability, 0.5)

    def test_winner_gains_and_loser_loses_elo(self) -> None:
        expected = expected_home_win_prob(1500, 1500)
        home_after, away_after = update_elo_ratings(1500, 1500, 1, expected)
        self.assertGreater(home_after, 1500)
        self.assertLess(away_after, 1500)

    def test_pre_game_elo_is_stored_before_update(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "season": 2023,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_win": 1,
                },
                {
                    "game_id": "g2",
                    "game_date": "2024-01-02",
                    "season": 2023,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_win": 0,
                },
            ]
        )
        with_elo = add_elo_features(games)
        self.assertAlmostEqual(with_elo.loc[0, "home_elo_pre"], 1500)
        self.assertAlmostEqual(with_elo.loc[0, "away_elo_pre"], 1500)
        self.assertNotAlmostEqual(with_elo.loc[1, "home_elo_pre"], 1500)

    def test_upcoming_elo_uses_current_ratings_without_update(self) -> None:
        completed = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "season": 2023,
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "home_win": 1,
                }
            ]
        )
        upcoming = pd.DataFrame(
            [
                {
                    "game_id": "g2",
                    "game_date": "2024-01-03",
                    "season": 2023,
                    "home_team_id": 1,
                    "away_team_id": 2,
                },
                {
                    "game_id": "g3",
                    "game_date": "2024-01-04",
                    "season": 2023,
                    "home_team_id": 1,
                    "away_team_id": 2,
                },
            ]
        )

        ratings, _ = current_elo_state(completed)
        features = add_upcoming_elo_features(completed, upcoming)

        self.assertAlmostEqual(features["home_elo_pre"].iloc[0], ratings[1])
        self.assertAlmostEqual(features["home_elo_pre"].iloc[1], ratings[1])


if __name__ == "__main__":
    unittest.main()
