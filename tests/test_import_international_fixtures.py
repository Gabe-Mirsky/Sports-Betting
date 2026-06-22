from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.fixtures_loader import (  # noqa: E402
    FIXTURE_COLUMNS,
    load_fixtures,
    normalize_fixtures,
    validate_fixtures,
)
from data.international_fixtures_importer import (  # noqa: E402
    make_fixture_id,
    normalize_international_fixtures,
)
from data.team_name_map import load_team_aliases  # noqa: E402


class TestInternationalFixturesImporter(unittest.TestCase):
    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-06-20",
                    "home_team": "Japan",
                    "away_team": "Tunisia",
                    "home_score": "",
                    "away_score": "",
                    "tournament": "Friendly",
                    "city": "Tokyo",
                    "country": "Japan",
                    "neutral": True,
                },
                {
                    "date": "2026-06-21",
                    "home_team": "USMNT",
                    "away_team": "Korea Republic",
                    "home_score": None,
                    "away_score": None,
                    "tournament": "FIFA World Cup",
                    "city": "New York",
                    "country": "United States",
                    "neutral": False,
                },
                {
                    "date": "2026-06-22",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "home_score": 2,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "city": "Rio de Janeiro",
                    "country": "Brazil",
                    "neutral": False,
                },
                {
                    "date": "2026-06-10",
                    "home_team": "Germany",
                    "away_team": "Sweden",
                    "home_score": None,
                    "away_score": None,
                    "tournament": "Friendly",
                    "city": "Berlin",
                    "country": "Germany",
                    "neutral": False,
                },
            ]
        )

    def _aliases(self, path: Path) -> dict[str, str]:
        path.write_text(
            "canonical_team,alias,sport,league,country\n"
            "United States,USMNT,soccer,international,United States\n"
            "South Korea,Korea Republic,soccer,international,South Korea\n",
            encoding="utf-8",
        )
        return load_team_aliases(path)

    def test_imports_future_fixtures_to_required_schema_without_odds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aliases = self._aliases(Path(tmp) / "aliases.csv")
            fixtures, stats = normalize_international_fixtures(
                self._raw(),
                aliases=aliases,
                as_of_date="2026-06-20",
                days_ahead=14,
            )

        self.assertEqual(list(fixtures.columns), FIXTURE_COLUMNS)
        self.assertEqual(int(stats["rows_read"]), 4)
        self.assertEqual(int(stats["rows_written"]), 2)
        self.assertEqual(int(stats["drop_reasons"]["completed_or_scored"]), 1)
        self.assertEqual(int(stats["drop_reasons"]["before_as_of_date"]), 1)
        self.assertNotIn("home_score", fixtures.columns)
        self.assertNotIn("away_score", fixtures.columns)
        self.assertFalse(any("odds" in c.lower() for c in fixtures.columns))

    def test_fixture_id_is_deterministic_and_safe(self) -> None:
        fixture_id = make_fixture_id("2026-06-20", "Japan", "Tunisia", "Friendly")
        self.assertEqual(fixture_id, "soccer_international_2026-06-20_japan_tunisia_friendly")
        self.assertNotRegex(fixture_id, r"[ /]")

    def test_keeps_future_missing_scores_and_drops_completed_scores(self) -> None:
        fixtures, _ = normalize_international_fixtures(self._raw(), as_of_date="2026-06-20")
        pairs = set(zip(fixtures["team_a"], fixtures["team_b"]))
        self.assertIn(("Japan", "Tunisia"), pairs)
        self.assertNotIn(("Brazil", "Argentina"), pairs)

    def test_neutral_site_clears_home_flags(self) -> None:
        fixtures, _ = normalize_international_fixtures(self._raw(), as_of_date="2026-06-20")
        row = fixtures[fixtures["team_a"] == "Japan"].iloc[0]
        self.assertEqual(int(row["neutral_site"]), 1)
        self.assertEqual(int(row["team_a_home_flag"]), 0)
        self.assertEqual(int(row["team_b_home_flag"]), 0)

    def test_applies_team_aliases_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aliases = self._aliases(Path(tmp) / "aliases.csv")
            fixtures, _ = normalize_international_fixtures(
                self._raw(),
                aliases=aliases,
                as_of_date="2026-06-20",
            )
        self.assertIn("United States", set(fixtures["team_a"]))
        self.assertIn("South Korea", set(fixtures["team_b"]))

    def test_output_loads_and_validates_with_fixture_loader(self) -> None:
        fixtures, _ = normalize_international_fixtures(self._raw(), as_of_date="2026-06-20")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures_today.csv"
            fixtures.to_csv(path, index=False)
            loaded = normalize_fixtures(load_fixtures(path))
        report = validate_fixtures(loaded)
        self.assertTrue(report["ok"], report)
        self.assertEqual(int(report["n_rows"]), len(fixtures))


if __name__ == "__main__":
    unittest.main()
