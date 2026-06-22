from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.current_basketball_join import (  # noqa: E402
    build_join_examples,
    build_join_report,
    join_zachht_to_current_games,
)


def _nba_games(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows: (game_id, date, home, away)."""

    frame = pd.DataFrame(rows, columns=["game_id", "game_date", "home_team_abbr", "away_team_abbr"])
    frame["home_score"] = 110
    frame["away_score"] = 100
    frame["canonical_game_key"] = [
        f"basketball|NBA|{d}|{h}|{a}" for _, d, h, a in rows
    ]
    return frame


def _zachht_games(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows: (game_ref, estimated_date, assumed_home, assumed_away)."""

    frame = pd.DataFrame(rows, columns=["source_game_id", "game_date", "home_team_abbr", "away_team_abbr"])
    frame["league"] = "NBA"
    frame["closing_snapshot_time"] = pd.Timestamp("2025-10-21 23:30:00")
    frame["n_snapshots"] = 5
    frame["canonical_game_key"] = [
        f"basketball|NBA|{d}|{h}|{a}" for _, d, h, a in rows
    ]
    return frame


class JoinZachhtToCurrentGamesTests(unittest.TestCase):
    def test_exact_join_on_canonical_key(self) -> None:
        nba = _nba_games([("001", "2025-10-21", "OKC", "HOU")])
        zachht = _zachht_games([("z1", "2025-10-21", "OKC", "HOU")])
        matched, unmatched_z, unmatched_n = join_zachht_to_current_games(zachht, nba)
        self.assertEqual(len(matched), 1)
        self.assertTrue(unmatched_z.empty)
        self.assertTrue(unmatched_n.empty)
        row = matched.iloc[0]
        self.assertEqual(row["date_offset_days"], 0)
        self.assertEqual(row["orientation"], "as_assumed")
        self.assertEqual(row["nba_game_id"], "001")

    def test_one_day_offset_is_matched_and_recorded(self) -> None:
        nba = _nba_games([("001", "2025-10-22", "OKC", "HOU")])
        zachht = _zachht_games([("z1", "2025-10-21", "OKC", "HOU")])
        matched, _, _ = join_zachht_to_current_games(zachht, nba)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["date_offset_days"], 1)

    def test_reversed_orientation_is_matched_and_recorded(self) -> None:
        nba = _nba_games([("001", "2025-10-21", "HOU", "OKC")])
        zachht = _zachht_games([("z1", "2025-10-21", "OKC", "HOU")])
        matched, _, _ = join_zachht_to_current_games(zachht, nba)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["orientation"], "reversed")

    def test_unmatched_rows_reported_on_both_sides(self) -> None:
        nba = _nba_games(
            [("001", "2025-10-21", "OKC", "HOU"), ("002", "2024-12-25", "LAL", "GSW")]
        )
        zachht = _zachht_games(
            [("z1", "2025-10-21", "OKC", "HOU"), ("z2", "2026-03-01", "BOS", "MIA")]
        )
        matched, unmatched_z, unmatched_n = join_zachht_to_current_games(zachht, nba)
        self.assertEqual(len(matched), 1)
        self.assertEqual(list(unmatched_z["source_game_id"]), ["z2"])
        self.assertEqual(list(unmatched_n["game_id"]), ["002"])

    def test_each_nba_game_claimed_at_most_once(self) -> None:
        nba = _nba_games([("001", "2025-10-21", "OKC", "HOU")])
        zachht = _zachht_games(
            [("z1", "2025-10-21", "OKC", "HOU"), ("z2", "2025-10-21", "OKC", "HOU")]
        )
        matched, unmatched_z, _ = join_zachht_to_current_games(zachht, nba)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(unmatched_z), 1)


class JoinReportTests(unittest.TestCase):
    def test_duplicate_canonical_keys_detected(self) -> None:
        nba = _nba_games(
            [("001", "2025-10-21", "OKC", "HOU"), ("002", "2025-10-21", "OKC", "HOU")]
        )
        zachht = _zachht_games([("z1", "2025-10-21", "OKC", "HOU")])
        matched, unmatched_z, unmatched_n = join_zachht_to_current_games(zachht, nba)
        report = build_join_report(matched, unmatched_z, unmatched_n, zachht, nba)
        self.assertEqual(report["duplicate_canonical_keys"]["nba_api"], 1)
        self.assertEqual(report["duplicate_canonical_keys"]["zachht"], 0)

    def test_grading_feasibility_flags(self) -> None:
        nba = _nba_games([("001", "2025-10-21", "OKC", "HOU")])
        zachht = _zachht_games([("z1", "2025-10-21", "OKC", "HOU")])
        matched, unmatched_z, unmatched_n = join_zachht_to_current_games(zachht, nba)
        report = build_join_report(matched, unmatched_z, unmatched_n, zachht, nba)
        self.assertTrue(report["grading_feasibility"]["settlement_grading_possible"])
        self.assertTrue(report["grading_feasibility"]["clv_grading_possible"])
        self.assertEqual(report["joined_games"], 1)
        self.assertTrue(report["research_only"])
        self.assertFalse(report["approved"])

    def test_examples_include_all_categories(self) -> None:
        nba = _nba_games(
            [
                ("001", "2025-10-21", "OKC", "HOU"),
                ("002", "2025-10-22", "LAL", "GSW"),
                ("003", "2024-12-25", "BOS", "MIA"),
            ]
        )
        zachht = _zachht_games(
            [
                ("z1", "2025-10-21", "OKC", "HOU"),  # exact
                ("z2", "2025-10-21", "LAL", "GSW"),  # +1 day
                ("z3", "2026-03-01", "DEN", "MIN"),  # unmatched
            ]
        )
        matched, unmatched_z, unmatched_n = join_zachht_to_current_games(zachht, nba)
        examples = build_join_examples(matched, unmatched_z, unmatched_n)
        categories = set(examples["category"])
        self.assertIn("joined_exact", categories)
        self.assertIn("joined_date_mismatch", categories)
        self.assertIn("unmatched_zachht", categories)
        self.assertIn("unmatched_nba_api", categories)


if __name__ == "__main__":
    unittest.main()
