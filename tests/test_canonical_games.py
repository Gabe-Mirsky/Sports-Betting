from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.canonical_games import (  # noqa: E402
    build_canonical_game_key,
    build_matchup_key,
    canonical_game_key_series,
    collapse_zachht_games,
    estimate_game_date_from_snapshot,
    normalize_team_for_key,
    parse_canonical_game_key,
    summarize_canonical_games,
)


class CanonicalGameKeyTests(unittest.TestCase):
    def test_basic_key_creation(self) -> None:
        key = build_canonical_game_key("basketball", "NBA", "2025-10-21", "OKC", "HOU")
        self.assertEqual(key, "basketball|NBA|2025-10-21|OKC|HOU")

    def test_key_normalizes_team_names_cities_and_alt_abbrs(self) -> None:
        variants = [
            ("Oklahoma City Thunder", "Houston Rockets"),
            ("THUNDER", "rockets"),
            ("okc", "Hou"),
        ]
        for home, away in variants:
            key = build_canonical_game_key("Basketball", "nba", "2025-10-21", home, away)
            self.assertEqual(key, "basketball|NBA|2025-10-21|OKC|HOU")

    def test_key_normalizes_sport_league_and_date_formats(self) -> None:
        key = build_canonical_game_key(" Basketball ", " nba ", "10/21/2025", "LAL", "GSW")
        self.assertEqual(key, "basketball|NBA|2025-10-21|LAL|GSW")

    def test_legacy_abbreviations_map_to_current(self) -> None:
        self.assertEqual(normalize_team_for_key("PHO", "NBA"), "PHX")
        self.assertEqual(normalize_team_for_key("BRK", "NBA"), "BKN")
        self.assertEqual(normalize_team_for_key("WSH", "NBA"), "WAS")
        self.assertEqual(normalize_team_for_key("Los Angeles Clippers", "NBA"), "LAC")

    def test_non_nba_league_uses_compact_text(self) -> None:
        self.assertEqual(normalize_team_for_key("Las Vegas Aces", "WNBA"), "LASVEGASACES")

    def test_missing_component_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_canonical_game_key("basketball", "NBA", None, "OKC", "HOU")
        with self.assertRaises(ValueError):
            build_canonical_game_key("basketball", "NBA", "not a date", "OKC", "HOU")
        with self.assertRaises(ValueError):
            build_canonical_game_key("basketball", "NBA", "2025-10-21", "", "HOU")

    def test_home_equals_away_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_canonical_game_key("basketball", "NBA", "2025-10-21", "OKC", "Thunder")

    def test_parse_roundtrip(self) -> None:
        key = build_canonical_game_key("basketball", "NBA", "2025-10-21", "OKC", "HOU")
        parts = parse_canonical_game_key(key)
        self.assertEqual(parts["home_team"], "OKC")
        self.assertEqual(parts["away_team"], "HOU")
        self.assertEqual(parts["game_date"], "2025-10-21")

    def test_matchup_key_is_order_independent(self) -> None:
        a = build_matchup_key("basketball", "NBA", "2025-10-21", "OKC", "HOU")
        b = build_matchup_key("basketball", "NBA", "2025-10-21", "Houston Rockets", "Thunder")
        self.assertEqual(a, b)
        self.assertEqual(a, "basketball|NBA|2025-10-21|HOU|OKC")

    def test_vectorized_series_blanks_bad_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "league": ["NBA", "NBA"],
                "game_date": ["2025-10-21", None],
                "home": ["OKC", "LAL"],
                "away": ["HOU", "GSW"],
            }
        )
        keys = canonical_game_key_series(frame, "basketball", "league", "game_date", "home", "away")
        self.assertEqual(keys.iloc[0], "basketball|NBA|2025-10-21|OKC|HOU")
        self.assertEqual(keys.iloc[1], "")


class ZachhtCollapseTests(unittest.TestCase):
    def test_estimate_game_date_converts_utc_to_eastern(self) -> None:
        # 02:30 UTC is the previous calendar day in US/Eastern.
        self.assertEqual(estimate_game_date_from_snapshot("2026-01-15 02:30:00"), "2026-01-14")
        self.assertEqual(estimate_game_date_from_snapshot("2025-10-21 23:30:00"), "2025-10-21")
        self.assertEqual(estimate_game_date_from_snapshot("garbage"), "")

    def test_collapse_uses_closing_snapshot_and_team2_as_home(self) -> None:
        snapshots = pd.DataFrame(
            {
                "league": ["NBA", "NBA"],
                "game_ref": ["g1", "g1"],
                "team1_name": ["Houston Rockets"] * 2,
                "team2_name": ["Oklahoma City Thunder"] * 2,
                "team1_abbr": ["HOU"] * 2,
                "team2_abbr": ["OKC"] * 2,
                "snapshot_time": ["2025-10-10 12:00:00", "2025-10-21 23:30:00"],
                "n_snapshots_for_game": [2, 2],
            }
        )
        games = collapse_zachht_games(snapshots)
        self.assertEqual(len(games), 1)
        row = games.iloc[0]
        self.assertEqual(row["home_team_abbr"], "OKC")
        self.assertEqual(row["away_team_abbr"], "HOU")
        self.assertEqual(row["game_date"], "2025-10-21")
        self.assertEqual(row["canonical_game_key"], "basketball|NBA|2025-10-21|OKC|HOU")
        self.assertFalse(bool(row["home_away_confident"]))
        self.assertTrue(bool(row["game_date_estimated"]))


class CanonicalSummaryTests(unittest.TestCase):
    def test_duplicate_keys_within_source_are_counted(self) -> None:
        table = pd.DataFrame(
            {
                "canonical_game_key": ["k1", "k1", "k2", "k2"],
                "matchup_key": ["m1", "m1", "m2", "m2"],
                "key_version": ["v1"] * 4,
                "sport": ["basketball"] * 4,
                "league": ["NBA"] * 4,
                "game_date": ["2025-10-21"] * 4,
                "home_team_abbr": ["OKC"] * 4,
                "away_team_abbr": ["HOU"] * 4,
                "source": ["nba_api", "nba_api", "nba_api", "zachht"],
                "source_game_id": ["1", "2", "3", "4"],
                "game_date_estimated": [False] * 4,
                "home_away_confident": [True] * 4,
            }
        )
        summary = summarize_canonical_games(table)
        self.assertEqual(summary["duplicate_keys_within_source"]["nba_api"], 1)
        self.assertEqual(summary["duplicate_keys_within_source"]["zachht"], 0)
        self.assertEqual(summary["cross_source_key_overlap"]["nba_api__zachht"], 1)


if __name__ == "__main__":
    unittest.main()
