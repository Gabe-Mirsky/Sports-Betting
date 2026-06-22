from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.nba_collection_planner import (  # noqa: E402
    COLLECTION_WINDOWS,
    build_clv_readiness_summary,
    build_collection_plan,
    build_coverage_frame,
    classify_timing,
    minutes_until_next_nba_game,
    write_collection_plan_reports,
)


NOW = datetime(2026, 6, 10, 19, 0, 0, tzinfo=timezone.utc)
GAME_KEY = "basketball|NBA|2026-06-10|SAS|NYK"


def _snapshot_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "league": "NBA",
        "sport": "basketball",
        "canonical_game_key": GAME_KEY,
        "player_name": "Test Player",
        "prop_type": "points",
        "line": 25.5,
        "bookmaker": "draftkings",
        "source": "odds_api",
        "is_closing_snapshot": False,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _snap(minutes_before_tip: float, start: datetime) -> dict:
    return {
        "snapshot_time": (start - timedelta(minutes=minutes_before_tip)).isoformat(),
        "game_start_time": start.isoformat(),
    }


class ClassifyTimingTests(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(classify_timing(-5), "post_start")
        self.assertEqual(classify_timing(30), "closing_like")
        self.assertEqual(classify_timing(90), "late")
        self.assertEqual(classify_timing(300), "mid")
        self.assertEqual(classify_timing(720), "early")
        self.assertEqual(classify_timing(2000), "very_early")


class PlanTests(unittest.TestCase):
    def test_windows_hit_missed_and_upcoming(self) -> None:
        # Game tips in 5 hours; snapshots at ~24h and ~10h before tip.
        start = NOW + timedelta(hours=5)
        snaps = _snapshot_frame([_snap(1440, start), _snap(600, start)])
        plan = build_collection_plan(snaps, now=NOW)

        self.assertEqual(plan["games_total"], 1)
        game = plan["games"][0]
        self.assertEqual(game["timing_classification"], "mid")
        # Both early snapshots land in the 24h band (360, 2880].
        self.assertIn("24h_before", game["windows_hit"])
        # 6h band (120, 360] has passed unhit? minutes_until_game=300 -> 6h band
        # still open (300 <= 360), so it is open_now, not missed.
        self.assertIn("6h_before", game["windows_open_now"])
        self.assertTrue(game["collection_needed_now"])
        self.assertIn("2h_before", [w["window"] for w in game["windows"] if w["window_status"] == "upcoming"])
        self.assertEqual(game["windows_missed"], [])
        self.assertTrue(game["clv_possible"])

    def test_missed_windows_after_tip(self) -> None:
        # Game already started; only one early snapshot exists.
        start = NOW - timedelta(minutes=30)
        snaps = _snapshot_frame([_snap(1200, start)])
        plan = build_collection_plan(snaps, now=NOW)

        game = plan["games"][0]
        self.assertEqual(game["timing_classification"], "post_start")
        self.assertIn("24h_before", game["windows_hit"])
        for window in ("6h_before", "2h_before", "60m_before", "30m_before", "10m_before"):
            self.assertIn(window, game["windows_missed"])
        self.assertFalse(game["collection_needed_now"])
        self.assertIsNone(game["next_recommended_collection_time_utc"])
        self.assertFalse(game["clv_possible"])
        self.assertTrue(any("NOT recoverable" in w for w in plan["warnings"]))

    def test_clv_capable_when_early_and_closing_exist(self) -> None:
        start = NOW - timedelta(minutes=30)
        snaps = _snapshot_frame([_snap(1200, start), _snap(45, start)])
        plan = build_collection_plan(snaps, now=NOW)
        game = plan["games"][0]
        self.assertEqual(game["closing_like_snapshots"], 1)
        self.assertTrue(game["clv_possible"])

        readiness = build_clv_readiness_summary(plan, snaps)
        self.assertTrue(readiness["clv_possible_now"])
        self.assertIn(GAME_KEY, readiness["games_clv_capable_now"])

    def test_empty_snapshots_warns(self) -> None:
        plan = build_collection_plan(pd.DataFrame(), now=NOW)
        self.assertEqual(plan["games_total"], 0)
        self.assertFalse(plan["collection_needed_now"])
        self.assertTrue(plan["warnings"])
        readiness = build_clv_readiness_summary(plan, pd.DataFrame())
        self.assertFalse(readiness["clv_possible_now"])
        self.assertFalse(readiness["clv_possible_later"])

    def test_minutes_until_next_nba_game(self) -> None:
        start = NOW + timedelta(minutes=200)
        snaps = _snapshot_frame([_snap(1000, start)])
        self.assertAlmostEqual(minutes_until_next_nba_game(snaps, NOW), 200.0, places=1)
        self.assertIsNone(minutes_until_next_nba_game(pd.DataFrame(), NOW))
        # Past games do not count.
        past = _snapshot_frame([_snap(60, NOW - timedelta(hours=2))])
        self.assertIsNone(minutes_until_next_nba_game(past, NOW))

    def test_coverage_frame_one_row_per_game_window(self) -> None:
        start = NOW + timedelta(hours=5)
        snaps = _snapshot_frame([_snap(1440, start)])
        plan = build_collection_plan(snaps, now=NOW)
        coverage = build_coverage_frame(plan)
        self.assertEqual(len(coverage), len(COLLECTION_WINDOWS))
        self.assertEqual(set(coverage["canonical_game_key"]), {GAME_KEY})
        self.assertEqual(
            set(coverage["window_status"]) - {"hit", "missed", "open_now", "upcoming"}, set()
        )


class WriteReportsTests(unittest.TestCase):
    def test_write_outputs(self) -> None:
        start = NOW + timedelta(hours=5)
        snaps = _snapshot_frame([_snap(1440, start), _snap(600, start)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            snaps.to_csv(processed / "player_prop_snapshots_normalized.csv", index=False)

            plan = write_collection_plan_reports(root, now=NOW)

            reports = root / "data" / "reports"
            for filename in (
                "nba_prop_closing_collection_plan.json",
                "nba_prop_closing_collection_plan.md",
                "nba_prop_closing_coverage.csv",
                "nba_prop_clv_readiness_summary.json",
            ):
                self.assertTrue((reports / filename).exists(), filename)
            loaded = json.loads(
                (reports / "nba_prop_closing_collection_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["games_total"], 1)
            self.assertTrue(plan["research_only"])
            readiness = json.loads(
                (reports / "nba_prop_clv_readiness_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(readiness["approved"])


class DashboardSectionTests(unittest.TestCase):
    def test_player_props_page_shows_clv_readiness(self) -> None:
        from reports.dashboard import write_static_dashboard_pages

        start = NOW + timedelta(hours=5)
        snaps = _snapshot_frame([_snap(1440, start), _snap(600, start)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            snaps.to_csv(processed / "player_prop_snapshots_normalized.csv", index=False)
            write_collection_plan_reports(root, now=NOW)

            reports_dir = root / "data" / "reports"
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            content = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", content)
            self.assertIn("Games Collected", content)
            self.assertIn("Complete Markets", content)
            self.assertIn("Advanced reports", content)
            self.assertIn("nba_prop_closing_coverage.csv", content)
            self.assertIn("nba_prop_clv_readiness_summary.json", content)
            self.assertNotIn("Missed Windows", content)

    def test_player_props_page_shows_new_report_sections(self) -> None:
        from reports.dashboard import write_static_dashboard_pages

        with tempfile.TemporaryDirectory() as folder:
            reports_dir = Path(folder)
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            content = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", content)
            self.assertIn("Research Bets We Would Place", content)
            self.assertIn("No qualifying research bets yet.", content)
            self.assertNotIn("Research Signals Status", content)

    def test_player_props_page_clv_section_empty_state(self) -> None:
        from reports.dashboard import write_static_dashboard_pages

        with tempfile.TemporaryDirectory() as folder:
            reports_dir = Path(folder)
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            content = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", content)
            self.assertIn("No qualifying research bets yet.", content)
            self.assertNotIn("build_nba_collection_plan.py", content)


if __name__ == "__main__":
    unittest.main()
