from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.nba_current_actuals import (  # noqa: E402
    normalize_current_games,
    normalize_current_player_logs,
)


def _team_log_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["0022500001", "0022500001"],
            "GAME_DATE": ["2025-10-21", "2025-10-21"],
            "TEAM_ID": [1610612760, 1610612745],
            "TEAM_ABBREVIATION": ["OKC", "HOU"],
            "TEAM_NAME": ["Oklahoma City Thunder", "Houston Rockets"],
            "MATCHUP": ["OKC vs. HOU", "HOU @ OKC"],
            "WL": ["W", "L"],
            "PTS": [125, 124],
            "nba_season": ["2025-26", "2025-26"],
            "season_type": ["Regular Season", "Regular Season"],
            "season_start_year": [2025, 2025],
        }
    )


def _player_log_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_ID": [101, 202],
            "PLAYER_NAME": ["Shai Gilgeous-Alexander", "Kevin Durant"],
            "TEAM_ID": [1610612760, 1610612745],
            "TEAM_ABBREVIATION": ["OKC", "HOU"],
            "TEAM_NAME": ["Oklahoma City Thunder", "Houston Rockets"],
            "GAME_ID": ["0022500001", "0022500001"],
            "GAME_DATE": ["2025-10-21", "2025-10-21"],
            "MATCHUP": ["OKC vs. HOU", "HOU @ OKC"],
            "MIN": [38, 36],
            "PTS": [35, 23],
            "REB": [5, 9],
            "AST": [5, 4],
            "FG3M": [2, 1],
            "STL": [1, 0],
            "BLK": [1, 1],
            "TOV": [3, 2],
            "nba_season": ["2025-26", "2025-26"],
            "season_type": ["Regular Season", "Regular Season"],
            "season_start_year": [2025, 2025],
        }
    )


class NormalizeCurrentGamesTests(unittest.TestCase):
    def test_pivot_to_one_row_per_game_with_canonical_key(self) -> None:
        games = normalize_current_games(_team_log_fixture())
        self.assertEqual(len(games), 1)
        row = games.iloc[0]
        self.assertEqual(row["home_team_abbr"], "OKC")
        self.assertEqual(row["away_team_abbr"], "HOU")
        self.assertEqual(row["home_score"], 125)
        self.assertEqual(row["away_score"], 124)
        self.assertEqual(row["winner"], "home")
        self.assertTrue(bool(row["home_win"]))
        self.assertEqual(row["canonical_game_key"], "basketball|NBA|2025-10-21|OKC|HOU")
        self.assertEqual(row["season"], "2025-26")

    def test_empty_input_returns_empty_frame(self) -> None:
        self.assertTrue(normalize_current_games(pd.DataFrame()).empty)


class NormalizeCurrentPlayerLogsTests(unittest.TestCase):
    def test_stats_and_keys_normalized(self) -> None:
        players = normalize_current_player_logs(_player_log_fixture())
        self.assertEqual(len(players), 2)
        home_row = players[players["player_name"].eq("Shai Gilgeous-Alexander")].iloc[0]
        away_row = players[players["player_name"].eq("Kevin Durant")].iloc[0]

        self.assertTrue(bool(home_row["is_home"]))
        self.assertFalse(bool(away_row["is_home"]))
        self.assertEqual(home_row["opponent_abbr"], "HOU")
        self.assertEqual(away_row["opponent_abbr"], "OKC")
        self.assertEqual(home_row["points"], 35)
        self.assertEqual(home_row["minutes"], 38)
        self.assertEqual(away_row["rebounds"], 9)
        self.assertEqual(away_row["threes"], 1)
        self.assertEqual(away_row["turnovers"], 2)

        # Both players in the same game share the same canonical key, oriented home/away.
        expected = "basketball|NBA|2025-10-21|OKC|HOU"
        self.assertEqual(home_row["canonical_game_key"], expected)
        self.assertEqual(away_row["canonical_game_key"], expected)

    def test_player_key_matches_game_key(self) -> None:
        games = normalize_current_games(_team_log_fixture())
        players = normalize_current_player_logs(_player_log_fixture())
        self.assertEqual(
            set(players["canonical_game_key"]), set(games["canonical_game_key"])
        )


if __name__ == "__main__":
    unittest.main()
