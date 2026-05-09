from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.team_aliases import normalize_team_abbr  # noqa: E402


class TestTeamAliases(unittest.TestCase):
    def test_common_short_aliases(self) -> None:
        self.assertEqual(normalize_team_abbr("NY"), "NYK")
        self.assertEqual(normalize_team_abbr("GS"), "GSW")
        self.assertEqual(normalize_team_abbr("SA"), "SAS")
        self.assertEqual(normalize_team_abbr("PHO"), "PHX")

    def test_team_names(self) -> None:
        self.assertEqual(normalize_team_abbr("Golden State Warriors"), "GSW")
        self.assertEqual(normalize_team_abbr("New York Knicks"), "NYK")
        self.assertEqual(normalize_team_abbr("LA Lakers"), "LAL")
        self.assertEqual(normalize_team_abbr("Cavs"), "CLE")

    def test_unknown_codes_are_preserved_compactly(self) -> None:
        self.assertEqual(normalize_team_abbr("AAA"), "AAA")


if __name__ == "__main__":
    unittest.main()
