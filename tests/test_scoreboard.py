from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.scoreboard import (  # noqa: E402
    fill_team_abbreviations_from_history,
    infer_season_from_date,
    infer_season_type_from_date,
    parse_scoreboard_frames,
    parse_scoreboard_v3_frames,
)


class TestScoreboard(unittest.TestCase):
    def test_infers_nba_season_and_playoffs(self) -> None:
        self.assertEqual(infer_season_from_date("2026-05-07"), 2025)
        self.assertEqual(infer_season_type_from_date("2026-05-07"), "Playoffs")
        self.assertEqual(infer_season_type_from_date("2025-12-25"), "Regular Season")

    def test_parse_scoreboard_frames_returns_model_columns(self) -> None:
        header = pd.DataFrame(
            {
                "GAME_DATE_EST": ["2026-05-07"],
                "GAME_ID": ["0042500201"],
                "GAME_STATUS_ID": [1],
                "GAME_STATUS_TEXT": ["8:00 pm ET"],
                "HOME_TEAM_ID": [1610612738],
                "VISITOR_TEAM_ID": [1610612752],
                "SEASON": [2025],
            }
        )
        lines = pd.DataFrame(
            {
                "GAME_ID": ["0042500201", "0042500201"],
                "TEAM_ID": [1610612738, 1610612752],
                "TEAM_ABBREVIATION": ["BOS", "NYK"],
            }
        )

        games = parse_scoreboard_frames(header, lines)

        self.assertEqual(games["home_team_abbr"].iloc[0], "BOS")
        self.assertEqual(games["away_team_abbr"].iloc[0], "NYK")
        self.assertEqual(games["season_type"].iloc[0], "Playoffs")

    def test_parse_scoreboard_v3_frames_uses_game_code_for_home_away(self) -> None:
        header = pd.DataFrame(
            {
                "gameId": ["0042500202"],
                "gameCode": ["20260507/CLEDET"],
                "gameStatus": [1],
                "gameStatusText": ["7:00 pm ET"],
                "gameEt": ["2026-05-07T19:00:00Z"],
                "poRoundDesc": ["Conf. Semifinals"],
            }
        )
        lines = pd.DataFrame(
            {
                "gameId": ["0042500202", "0042500202"],
                "teamId": [1610612765, 1610612739],
                "teamTricode": ["DET", "CLE"],
            }
        )

        games = parse_scoreboard_v3_frames(header, lines)

        self.assertEqual(games["home_team_abbr"].iloc[0], "DET")
        self.assertEqual(games["away_team_abbr"].iloc[0], "CLE")
        self.assertEqual(int(games["home_team_id"].iloc[0]), 1610612765)
        self.assertEqual(games["season_type"].iloc[0], "Playoffs")

    def test_fill_team_abbreviations_from_history(self) -> None:
        upcoming = pd.DataFrame(
            {
                "home_team_id": [1],
                "away_team_id": [2],
                "home_team_abbr": [pd.NA],
                "away_team_abbr": [pd.NA],
            }
        )
        historical = pd.DataFrame(
            {
                "home_team_id": [1],
                "home_team_abbr": ["AAA"],
                "away_team_id": [2],
                "away_team_abbr": ["BBB"],
            }
        )

        filled = fill_team_abbreviations_from_history(upcoming, historical)

        self.assertEqual(filled["home_team_abbr"].iloc[0], "AAA")
        self.assertEqual(filled["away_team_abbr"].iloc[0], "BBB")


if __name__ == "__main__":
    unittest.main()
