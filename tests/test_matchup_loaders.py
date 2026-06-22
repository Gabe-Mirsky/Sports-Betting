from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.fixtures_loader import normalize_fixtures, validate_fixtures  # noqa: E402
from data.injuries_loader import (  # noqa: E402
    normalize_injuries,
    summarize_team_availability,
)
from data.match_results_loader import (  # noqa: E402
    MATCH_RESULTS_COLUMNS,
    normalize_match_results,
    validate_match_results,
)
from data.team_name_map import normalize_team_name, team_match_key  # noqa: E402


class TestMatchResultsLoader(unittest.TestCase):
    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": "2026-01-01", "home_team": "Japan", "away_team": "Korea Republic",
                 "home_score": 2, "away_score": 2, "sport": "soccer", "league": "intl",
                 "competition": "friendly"},
                {"date": "2026-01-05", "home_team": "Brazil", "away_team": "Japan National Team",
                 "home_score": 1, "away_score": 0, "sport": "soccer", "league": "intl",
                 "competition": "qualifier"},
            ]
        )

    def test_normalized_schema_and_results(self) -> None:
        out = normalize_match_results(self._raw())
        for column in MATCH_RESULTS_COLUMNS:
            self.assertIn(column, out.columns)
        # Draw recorded for soccer when scores are equal.
        draw_row = out[out["team_b"] == "South Korea"].iloc[0]
        self.assertEqual(int(draw_row["result_draw"]), 1)
        self.assertEqual(int(draw_row["result_team_a_win"]), 0)
        # Decisive game.
        win_row = out[out["team_a"] == "Brazil"].iloc[0]
        self.assertEqual(int(win_row["result_team_a_win"]), 1)

    def test_team_aliases_normalized(self) -> None:
        out = normalize_match_results(self._raw())
        self.assertIn("South Korea", set(out["team_b"]))
        # "Japan National Team" collapses to "Japan".
        self.assertIn("Japan", set(out["team_b"]).union(out["team_a"]))

    def test_no_draw_sport_forces_zero_draw(self) -> None:
        raw = pd.DataFrame(
            [{"date": "2026-01-01", "home_team": "LAL", "away_team": "BOS",
              "home_score": 100, "away_score": 99, "sport": "basketball", "league": "nba"}]
        )
        out = normalize_match_results(raw, {"default_sport": "basketball"})
        self.assertEqual(int(out["result_draw"].iloc[0]), 0)

    def test_validate_reports_ok(self) -> None:
        report = validate_match_results(normalize_match_results(self._raw()))
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_rows"], 2)


class TestFixturesLoader(unittest.TestCase):
    def test_neutral_site_clears_home_flags(self) -> None:
        raw = pd.DataFrame(
            [{"date": "2026-06-20", "team_a": "Japan", "team_b": "Tunisia",
              "neutral": True, "sport": "soccer", "league": "intl",
              "competition_type": "friendly"}]
        )
        out = normalize_fixtures(raw)
        row = out.iloc[0]
        self.assertEqual(int(row["neutral_site"]), 1)
        self.assertEqual(int(row["team_a_home_flag"]), 0)
        self.assertEqual(int(row["team_b_home_flag"]), 0)
        self.assertTrue(validate_fixtures(out)["ok"])


class TestInjuriesLoader(unittest.TestCase):
    def test_importance_fallback_from_role(self) -> None:
        raw = pd.DataFrame(
            [
                {"team": "T", "player": "Star", "status": "out", "role": "star"},
                {"team": "T", "player": "Sub", "status": "out", "role": "bench"},
            ]
        )
        out = normalize_injuries(raw)
        self.assertAlmostEqual(out.iloc[0]["importance_score"], 1.0)
        self.assertAlmostEqual(out.iloc[1]["importance_score"], 0.15)

    def test_summary_counts_and_staleness(self) -> None:
        raw = pd.DataFrame(
            [
                {"team": "T", "player": "Star", "status": "out", "role": "star",
                 "last_updated": "2026-06-19T10:00:00"},
                {"team": "T", "player": "Sub", "status": "questionable", "role": "bench",
                 "last_updated": "2026-06-19T10:00:00"},
            ]
        )
        summary = summarize_team_availability(normalize_injuries(raw), "2026-06-20T10:00:00")
        row = summary.iloc[0]
        self.assertEqual(int(row["players_out"]), 1)
        self.assertEqual(int(row["key_players_out"]), 1)
        self.assertFalse(bool(row["injury_data_stale"]))  # 24h old < 48h

    def test_missing_timestamp_is_stale(self) -> None:
        raw = pd.DataFrame([{"team": "T", "player": "X", "status": "out", "role": "star"}])
        summary = summarize_team_availability(normalize_injuries(raw), "2026-06-20")
        self.assertTrue(bool(summary.iloc[0]["injury_data_stale"]))


class TestTeamNameMap(unittest.TestCase):
    def test_match_key_collapses_variants(self) -> None:
        self.assertEqual(team_match_key("Côte d'Ivoire"), team_match_key("Cote d Ivoire"))

    def test_normalize_known_aliases(self) -> None:
        self.assertEqual(normalize_team_name("USA"), "United States")
        self.assertEqual(normalize_team_name("Korea Republic"), "South Korea")


if __name__ == "__main__":
    unittest.main()
