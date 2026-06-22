from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.availability_template_builder import (  # noqa: E402
    AVAILABILITY_TEMPLATE_COLUMNS,
    build_availability_template_from_fixtures,
)


class TestAvailabilityTemplateBuilder(unittest.TestCase):
    def _fixtures(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fixture_id": "fx1",
                    "game_date": "2026-06-20",
                    "team_a": "Japan",
                    "team_b": "Tunisia",
                    "sport": "soccer",
                    "league": "international",
                    "competition_type": "FIFA World Cup",
                }
            ]
        )

    def test_creates_two_team_rows_per_fixture(self) -> None:
        template = build_availability_template_from_fixtures(self._fixtures(), as_of_date="2026-06-20")

        self.assertEqual(len(template), 2)
        self.assertEqual(list(template.columns), AVAILABILITY_TEMPLATE_COLUMNS)
        self.assertEqual(set(template["team"]), {"Japan", "Tunisia"})
        japan = template[template["team"] == "Japan"].iloc[0]
        self.assertEqual(japan["fixture_id"], "fx1")
        self.assertEqual(japan["opponent"], "Tunisia")
        self.assertEqual(japan["game_date"], "2026-06-20")
        self.assertEqual(japan["status"], "unknown")
        self.assertEqual(japan["source"], "manual")

    def test_avoids_duplicate_fixture_team_player_rows(self) -> None:
        fixtures = pd.concat([self._fixtures(), self._fixtures()], ignore_index=True)

        template = build_availability_template_from_fixtures(fixtures, as_of_date="2026-06-20")

        self.assertEqual(len(template), 2)
        self.assertFalse(template.duplicated(subset=["fixture_id", "team", "player_name"]).any())

    def test_supports_placeholder_players(self) -> None:
        template = build_availability_template_from_fixtures(
            self._fixtures(),
            as_of_date="2026-06-20",
            include_placeholder_players=True,
            players_per_team=2,
        )

        self.assertEqual(len(template), 4)
        self.assertEqual(
            set(template["player_name"]),
            {"Unknown player 1", "Unknown player 2"},
        )


if __name__ == "__main__":
    unittest.main()
