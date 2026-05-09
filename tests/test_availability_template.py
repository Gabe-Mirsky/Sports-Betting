from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.availability_template import build_availability_template  # noqa: E402


class TestAvailabilityTemplate(unittest.TestCase):
    def test_template_uses_prior_player_minutes_for_impact_weights(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "prev",
                    "game_date": "2026-01-01",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                },
                {
                    "game_id": "next",
                    "game_date": "2026-01-10",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                },
            ]
        )
        player_logs = pd.DataFrame(
            [
                {
                    "GAME_ID": "prev",
                    "GAME_DATE": "2026-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "BOS",
                    "PLAYER_ID": 101,
                    "PLAYER_NAME": "Starter",
                    "MIN": 36,
                    "PTS": 20,
                    "REB": 5,
                    "AST": 5,
                    "STL": 1,
                    "BLK": 1,
                    "TOV": 2,
                    "PLUS_MINUS": 10,
                    "SEASON_ID": 22025,
                },
                {
                    "GAME_ID": "prev",
                    "GAME_DATE": "2026-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "BOS",
                    "PLAYER_ID": 102,
                    "PLAYER_NAME": "Bench",
                    "MIN": 12,
                    "PTS": 4,
                    "REB": 2,
                    "AST": 1,
                    "STL": 0,
                    "BLK": 0,
                    "TOV": 1,
                    "PLUS_MINUS": -3,
                    "SEASON_ID": 22025,
                },
            ]
        )

        template = build_availability_template(
            games,
            player_logs,
            start_date="2026-01-10",
            end_date="2026-01-10",
            players_per_team=1,
        )

        self.assertEqual(len(template), 1)
        self.assertEqual(template.loc[0, "player_name"], "Starter")
        self.assertEqual(template.loc[0, "impact_weight"], 36)
        self.assertEqual(template.loc[0, "status"], "")


if __name__ == "__main__":
    unittest.main()
