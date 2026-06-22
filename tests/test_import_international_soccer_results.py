from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.international_soccer_importer import (  # noqa: E402
    OUTPUT_COLUMNS,
    make_game_id,
    normalize_international_results,
)
from data.match_results_loader import (  # noqa: E402
    MATCH_RESULTS_COLUMNS,
    load_match_results,
    normalize_match_results,
    validate_match_results,
)
from features.matchup_features import build_training_features  # noqa: E402
from quality.matchup_data_quality import validate_matchup_training_data  # noqa: E402

TODAY = pd.Timestamp("2026-06-20")

# Raw frame in the martj42/international_results layout.
_RAW = pd.DataFrame(
    [
        {"date": "2024-06-11", "home_team": "Japan", "away_team": "Tunisia",
         "home_score": 2, "away_score": 1, "tournament": "Friendly", "city": "Tokyo",
         "country": "Japan", "neutral": "FALSE"},
        {"date": "2024-06-15", "home_team": "Tunisia", "away_team": "Morocco",
         "home_score": 0, "away_score": 2, "tournament": "Friendly", "city": "Tunis",
         "country": "Tunisia", "neutral": "FALSE"},
        {"date": "2024-06-20", "home_team": "Brazil", "away_team": "Argentina",
         "home_score": 1, "away_score": 1, "tournament": "Copa America", "city": "Miami",
         "country": "United States", "neutral": "TRUE"},
        {"date": "2024-07-01", "home_team": "Spain", "away_team": "France",
         "home_score": "NA", "away_score": "NA", "tournament": "UEFA Euro", "city": "Berlin",
         "country": "Germany", "neutral": "FALSE"},
        {"date": "not-a-date", "home_team": "Italy", "away_team": "Germany",
         "home_score": 2, "away_score": 0, "tournament": "Friendly", "city": "Rome",
         "country": "Italy", "neutral": "FALSE"},
        {"date": "2030-01-01", "home_team": "England", "away_team": "Wales",
         "home_score": 1, "away_score": 0, "tournament": "Friendly", "city": "London",
         "country": "England", "neutral": "FALSE"},
    ]
)


def _clean() -> tuple[pd.DataFrame, dict]:
    return normalize_international_results(_RAW, today=TODAY)


class TestImporter(unittest.TestCase):
    def test_imports_sample_correctly(self) -> None:
        clean, stats = _clean()
        # 3 valid completed games; 3 dropped (bad score, bad date, future).
        self.assertEqual(len(clean), 3)
        self.assertEqual(stats["rows_read"], 6)
        self.assertEqual(stats["rows_written"], 3)

    def test_creates_required_project_columns(self) -> None:
        clean, _ = _clean()
        for column in OUTPUT_COLUMNS:
            self.assertIn(column, clean.columns)
        for column in MATCH_RESULTS_COLUMNS:
            self.assertIn(column, clean.columns)

    def test_maps_home_away_scores_to_team_a_b(self) -> None:
        clean, _ = _clean()
        row = clean[clean["team_a"] == "Japan"].iloc[0]
        self.assertEqual(row["team_b"], "Tunisia")
        self.assertEqual(int(row["team_a_score"]), 2)
        self.assertEqual(int(row["team_b_score"]), 1)

    def test_result_flags_home_away_draw(self) -> None:
        clean, _ = _clean()
        home_win = clean[clean["team_a"] == "Japan"].iloc[0]
        away_win = clean[clean["team_a"] == "Tunisia"].iloc[0]
        draw = clean[clean["team_a"] == "Brazil"].iloc[0]
        self.assertEqual((int(home_win["result_team_a_win"]), int(home_win["result_draw"]), int(home_win["result_team_b_win"])), (1, 0, 0))
        self.assertEqual((int(away_win["result_team_a_win"]), int(away_win["result_draw"]), int(away_win["result_team_b_win"])), (0, 0, 1))
        self.assertEqual((int(draw["result_team_a_win"]), int(draw["result_draw"]), int(draw["result_team_b_win"])), (0, 1, 0))

    def test_neutral_site_handling(self) -> None:
        clean, _ = _clean()
        neutral = clean[clean["team_a"] == "Brazil"].iloc[0]
        self.assertEqual(int(neutral["neutral_site"]), 1)
        self.assertEqual(int(neutral["team_a_home_flag"]), 0)
        home = clean[clean["team_a"] == "Japan"].iloc[0]
        self.assertEqual(int(home["neutral_site"]), 0)
        self.assertEqual(int(home["team_a_home_flag"]), 1)

    def test_drops_missing_scores(self) -> None:
        _, stats = _clean()
        self.assertGreaterEqual(stats["drop_reasons"].get("missing_or_bad_score", 0), 1)
        # The Spain/France NA row must not appear.
        clean, _ = _clean()
        self.assertNotIn("Spain", set(clean["team_a"]))

    def test_drops_bad_dates(self) -> None:
        _, stats = _clean()
        self.assertGreaterEqual(stats["drop_reasons"].get("bad_date", 0), 1)

    def test_drops_future_games(self) -> None:
        clean, stats = _clean()
        self.assertGreaterEqual(stats["drop_reasons"].get("future_or_after_today", 0), 1)
        self.assertNotIn("England", set(clean["team_a"]))

    def test_deterministic_game_id(self) -> None:
        clean1, _ = _clean()
        clean2, _ = _clean()
        self.assertEqual(list(clean1["game_id"]), list(clean2["game_id"]))
        expected = make_game_id("2024-06-11", "Japan", "Tunisia", 2, 1, "Friendly")
        self.assertEqual(expected, "soccer_international_2024-06-11_japan_tunisia_2_1_friendly")
        self.assertIn(expected, set(clean1["game_id"]))

    def test_no_odds_columns_required_or_present(self) -> None:
        clean, _ = _clean()
        odds_like = {"odds", "price", "clv", "implied_prob", "american_odds",
                     "decimal_odds", "closing_line", "vig", "no_vig_prob", "spread"}
        self.assertEqual(odds_like & {c.lower() for c in clean.columns}, set())

    def test_output_loadable_by_match_results_loader(self) -> None:
        clean, _ = _clean()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match_results.csv"
            clean.to_csv(path, index=False)
            normalized = normalize_match_results(load_match_results(path))
            for column in MATCH_RESULTS_COLUMNS:
                self.assertIn(column, normalized.columns)
            self.assertEqual(len(normalized), 3)

    def test_output_passes_validators(self) -> None:
        clean, _ = _clean()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match_results.csv"
            clean.to_csv(path, index=False)
            normalized = normalize_match_results(load_match_results(path))
        self.assertTrue(validate_match_results(normalized)["ok"])
        training = build_training_features(normalized)
        self.assertTrue(validate_matchup_training_data(training)["ok"])

    def test_exclude_friendlies(self) -> None:
        clean, stats = normalize_international_results(_RAW, today=TODAY, include_friendlies=False)
        # Japan/Tunisia and Tunisia/Morocco are friendlies -> only the Copa America draw remains.
        self.assertEqual(set(clean["competition_type"]), {"Copa America"})
        self.assertGreaterEqual(stats["drop_reasons"].get("friendlies_excluded", 0), 1)


if __name__ == "__main__":
    unittest.main()
