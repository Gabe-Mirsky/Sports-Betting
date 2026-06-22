"""Tests for the NBA prop review exports (synthetic fixtures only).

Covers the Phase-20 fixture set for review exports: a settled prop, an
alternate-line row, a missing-price row, and a bookmaker comparison spread.
Synthetic data is used exclusively in tests, never in real reports.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_nba_prop_review_exports import (  # noqa: E402
    annotate,
    build_alt_lines_review,
    build_bookmaker_comparison,
    build_main_lines_review,
    build_prop_board,
)


def _enriched_fixture() -> pd.DataFrame:
    """Synthetic NBA snapshots: settled main line, alt line, missing price."""
    base = {
        "league": "NBA",
        "game_date": "2026-06-10",
        "canonical_game_key": "basketball|NBA|2026-06-10|NYK|SAS",
        "team": "SAS",
        "prop_type": "points",
        "settlement_status": "settled",
        "actual_stat_value": 30.0,
        "over_won": True,
        "under_won": False,
        "push": False,
        "is_closing_snapshot": False,
    }
    rows = [
        # Main line, two snapshots (latest must win).
        {**base, "player_name": "Test Player", "bookmaker": "bookA", "line": 25.5,
         "over_price": 1.90, "under_price": 1.90, "snapshot_time": "2026-06-10T15:00:00+00:00"},
        {**base, "player_name": "Test Player", "bookmaker": "bookA", "line": 25.5,
         "over_price": 1.95, "under_price": 1.85, "snapshot_time": "2026-06-10T19:00:00+00:00",
         "is_closing_snapshot": True},
        # Alternate line on the same market.
        {**base, "player_name": "Test Player", "bookmaker": "bookA", "line": 30.5,
         "over_price": 3.20, "under_price": 1.30, "snapshot_time": "2026-06-10T19:00:00+00:00"},
        # Second bookmaker quoting a different main line (comparison spread).
        {**base, "player_name": "Test Player", "bookmaker": "bookB", "line": 26.5,
         "over_price": 2.00, "under_price": 1.80, "snapshot_time": "2026-06-10T19:00:00+00:00"},
        # Missing-price row (under absent) on its own market.
        {**base, "player_name": "Bench Player", "bookmaker": "bookA", "line": 10.5,
         "over_price": 1.87, "under_price": None, "snapshot_time": "2026-06-10T19:00:00+00:00",
         "settlement_status": "pending_result", "actual_stat_value": None,
         "over_won": None, "under_won": None, "push": None},
    ]
    return pd.DataFrame(rows)


def _quality_fixture() -> pd.DataFrame:
    rows = [
        {"league": "NBA", "player_name": "Test Player", "prop_type": "points",
         "bookmaker": "bookA", "canonical_game_key": "basketball|NBA|2026-06-10|NYK|SAS",
         "likely_main_line": 25.5, "line_quality_label": "main_plus_alt_lines"},
        {"league": "NBA", "player_name": "Test Player", "prop_type": "points",
         "bookmaker": "bookB", "canonical_game_key": "basketball|NBA|2026-06-10|NYK|SAS",
         "likely_main_line": 26.5, "line_quality_label": "clean"},
        {"league": "NBA", "player_name": "Bench Player", "prop_type": "points",
         "bookmaker": "bookA", "canonical_game_key": "basketball|NBA|2026-06-10|NYK|SAS",
         "likely_main_line": 10.5, "line_quality_label": "missing_prices"},
    ]
    return pd.DataFrame(rows)


class AnnotateTests(unittest.TestCase):
    def test_alt_line_flagged(self) -> None:
        frame = annotate(_enriched_fixture(), _quality_fixture())
        alt_rows = frame[frame["is_alt_line"]]
        self.assertEqual(len(alt_rows), 1)
        self.assertEqual(float(alt_rows.iloc[0]["line"]), 30.5)

    def test_main_lines_keep_latest_snapshot(self) -> None:
        frame = annotate(_enriched_fixture(), _quality_fixture())
        review = build_main_lines_review(frame)
        book_a = review[
            (review["player_name"] == "Test Player") & (review["bookmaker"] == "bookA")
        ]
        self.assertEqual(len(book_a), 1)
        self.assertEqual(float(book_a.iloc[0]["over_price"]), 1.95)
        self.assertTrue(bool(book_a.iloc[0]["is_closing_snapshot"]))
        # The settled outcome travels with the row.
        self.assertEqual(book_a.iloc[0]["settlement_status"], "settled")

    def test_alt_review_contains_only_alt_lines(self) -> None:
        frame = annotate(_enriched_fixture(), _quality_fixture())
        review = build_alt_lines_review(frame)
        self.assertEqual(len(review), 1)
        self.assertTrue(bool(review.iloc[0]["is_alt_line"]))

    def test_missing_price_row_survives(self) -> None:
        frame = annotate(_enriched_fixture(), _quality_fixture())
        board = build_prop_board(frame)
        bench = board[board["player_name"] == "Bench Player"]
        self.assertEqual(len(bench), 1)
        self.assertTrue(pd.isna(bench.iloc[0]["under_price"]))

    def test_bookmaker_comparison_spread(self) -> None:
        frame = annotate(_enriched_fixture(), _quality_fixture())
        comparison = build_bookmaker_comparison(frame)
        player = comparison[comparison["player_name"] == "Test Player"]
        self.assertEqual(set(player["bookmaker"]), {"bookA", "bookB"})
        self.assertTrue((player["line_disagreement"] == 1.0).all())
        self.assertTrue((player["bookmakers"] == 2).all())


if __name__ == "__main__":
    unittest.main()
