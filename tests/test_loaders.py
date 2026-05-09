from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.loaders import build_game_level_dataset  # noqa: E402


class TestLoaders(unittest.TestCase):
    def test_neutral_site_double_away_matchup_is_kept_and_marked(self) -> None:
        logs = pd.DataFrame(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DATE": "2025-01-01",
                    "TEAM_ID": 1,
                    "TEAM_ABBREVIATION": "AAA",
                    "MATCHUP": "AAA @ BBB",
                    "PTS": 100,
                    "PLUS_MINUS": -5,
                },
                {
                    "GAME_ID": "g1",
                    "GAME_DATE": "2025-01-01",
                    "TEAM_ID": 2,
                    "TEAM_ABBREVIATION": "BBB",
                    "MATCHUP": "BBB @ AAA",
                    "PTS": 105,
                    "PLUS_MINUS": 5,
                },
            ]
        )

        games = build_game_level_dataset(logs)

        self.assertEqual(len(games), 1)
        self.assertEqual(games.loc[0, "home_team_abbr"], "BBB")
        self.assertEqual(games.loc[0, "away_team_abbr"], "AAA")
        self.assertEqual(games.loc[0, "neutral_site"], 1)
        self.assertEqual(games.loc[0, "home_away_quality"], "neutral_site_inferred")


if __name__ == "__main__":
    unittest.main()
