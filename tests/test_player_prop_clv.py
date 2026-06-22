from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.player_prop_clv import (  # noqa: E402
    build_clv_frame,
    build_clv_summary,
    write_clv_reports,
)


GAME_KEY = "basketball|NBA|2026-06-10|NYK|SAS"


GAME_START = pd.Timestamp("2026-06-11T00:40:00+00:00")


def _row(minutes: float, *, closing: bool, line: float = 25.5,
         over: float | None = 1.91, under: float | None = 1.91, **overrides) -> dict:
    row = {
        "snapshot_time": (GAME_START - pd.Timedelta(minutes=minutes)).isoformat(),
        "league": "NBA",
        "sport": "basketball",
        "canonical_game_key": GAME_KEY,
        "game_date": "2026-06-10",
        "player_name": "Test Player",
        "prop_type": "points",
        "line": line,
        "over_price": over,
        "under_price": under,
        "bookmaker": "draftkings",
        "source": "odds_api",
        "is_closing_snapshot": closing,
        "minutes_to_game_start": minutes,
    }
    row.update(overrides)
    return row


class ClvFrameTests(unittest.TestCase):
    def test_same_line_clv_computed(self) -> None:
        snaps = pd.DataFrame(
            [
                _row(1440, closing=False, over=2.00, under=1.85),
                _row(45, closing=True, over=1.80, under=2.05),
            ]
        )
        clv = build_clv_frame(snaps)
        self.assertEqual(len(clv), 1)
        record = clv.iloc[0]
        self.assertTrue(record["price_clv_comparable"])
        self.assertEqual(record["line_move_direction"], "flat")
        # Over: bet early at 2.00, closed 1.80 -> beat the close by 2.00/1.80-1.
        self.assertAlmostEqual(record["clv_over_pct"], 2.00 / 1.80 - 1, places=4)
        # Under: early 1.85 vs close 2.05 -> negative CLV.
        self.assertAlmostEqual(record["clv_under_pct"], 1.85 / 2.05 - 1, places=4)
        self.assertEqual(record["over_price_move_direction"], "down")
        self.assertEqual(record["under_price_move_direction"], "up")

    def test_line_change_blocks_price_clv(self) -> None:
        snaps = pd.DataFrame(
            [
                _row(1440, closing=False, line=25.5, over=1.91, under=1.91),
                _row(30, closing=True, line=27.5, over=1.91, under=1.91),
            ]
        )
        clv = build_clv_frame(snaps)
        record = clv.iloc[0]
        self.assertFalse(record["price_clv_comparable"])
        self.assertEqual(record["line_move_direction"], "up")
        self.assertEqual(record["line_move"], 2.0)
        self.assertTrue(pd.isna(record["clv_over_pct"]))
        self.assertTrue(pd.isna(record["clv_under_pct"]))

    def test_no_closing_snapshot_yields_no_clv(self) -> None:
        snaps = pd.DataFrame([_row(1440, closing=False), _row(600, closing=False)])
        clv = build_clv_frame(snaps)
        self.assertTrue(clv.empty)

    def test_markets_isolated_by_bookmaker(self) -> None:
        snaps = pd.DataFrame(
            [
                _row(1440, closing=False),
                _row(45, closing=True),
                _row(1440, closing=False, bookmaker="fanduel"),
                # fanduel never got a closing snapshot -> no fanduel CLV row.
            ]
        )
        clv = build_clv_frame(snaps)
        self.assertEqual(len(clv), 1)
        self.assertEqual(clv.iloc[0]["bookmaker"], "draftkings")


class SummaryTests(unittest.TestCase):
    def test_not_ready_summary_explains_missing_data(self) -> None:
        snaps = pd.DataFrame([_row(1440, closing=False)])
        clv = build_clv_frame(snaps)
        summary = build_clv_summary(snaps, clv, next_collection_hint="2026-06-10T23:40:00+00:00")
        self.assertFalse(summary["clv_ready"])
        self.assertIn("NOT READY", summary["verdict"])
        self.assertTrue(any("closing-like" in w for w in summary["warnings"]))
        self.assertTrue(any("2026-06-10T23:40" in w for w in summary["warnings"]))
        self.assertTrue(summary["settlement_not_required"])
        self.assertFalse(summary["approved"])

    def test_ready_summary_counts(self) -> None:
        snaps = pd.DataFrame(
            [
                _row(1440, closing=False, over=2.00, under=1.85),
                _row(45, closing=True, over=1.80, under=2.05),
            ]
        )
        clv = build_clv_frame(snaps)
        summary = build_clv_summary(snaps, clv)
        self.assertTrue(summary["clv_ready"])
        self.assertEqual(summary["markets_with_clv"], 1)
        self.assertEqual(summary["nba_markets_with_clv"], 1)
        self.assertEqual(summary["price_clv_comparable_markets"], 1)
        self.assertIsNotNone(summary["avg_clv_over_pct"])


class WriteReportsTests(unittest.TestCase):
    def test_writes_all_outputs_when_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame([_row(1440, closing=False)]).to_csv(
                processed / "player_prop_snapshots_normalized.csv", index=False
            )

            summary = write_clv_reports(root)

            reports = root / "data" / "reports"
            for filename in (
                "player_prop_clv_summary.json",
                "player_prop_clv.csv",
                "player_prop_clv_by_bookmaker.csv",
                "player_prop_clv_by_prop_type.csv",
                "player_prop_clv.md",
            ):
                self.assertTrue((reports / filename).exists(), filename)
            self.assertFalse(summary["clv_ready"])
            md = (reports / "player_prop_clv.md").read_text(encoding="utf-8")
            self.assertIn("NOT READY", md)
            loaded = json.loads(
                (reports / "player_prop_clv_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(loaded["approved"])


if __name__ == "__main__":
    unittest.main()
