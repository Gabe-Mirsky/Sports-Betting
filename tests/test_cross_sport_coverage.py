"""Tests for the cross-sport collection coverage audit (synthetic fixtures).

Covers: every sport group detected, inactive / quota-skipped / not-configured
league classification, and the dashboard section render.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_cross_sport_collection_coverage import (  # noqa: E402
    EXPECTED_SPORT_GROUPS,
    build_coverage,
    league_status,
    render_markdown,
)


NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


def _league_cfg(sport: str, priority: int, sport_key: str, *, modeling: bool = False) -> dict:
    return {
        "sport": sport,
        "enabled": True,
        "modeling_priority": modeling,
        "collect_only": not modeling,
        "priority": priority,
        "max_events_per_run": 3,
        "sources": {"odds_api": {"sport_key": sport_key, "markets": {"player_points": "points"}}},
    }


def _config(*, omit: set[str] | None = None) -> dict:
    omit = omit or set()
    leagues = {
        "NBA": _league_cfg("basketball", 1, "basketball_nba", modeling=True),
        "WNBA": _league_cfg("basketball", 2, "basketball_wnba"),
        "NCAAB": _league_cfg("basketball", 6, "basketball_ncaab"),
        "MLB": _league_cfg("baseball", 3, "baseball_mlb"),
        "NHL": _league_cfg("hockey", 5, "icehockey_nhl"),
        "NFL": _league_cfg("football", 4, "americanfootball_nfl"),
        "NCAAF": _league_cfg("football", 13, "americanfootball_ncaaf"),
        "EPL": _league_cfg("soccer", 7, "soccer_epl"),
        "MLS": _league_cfg("soccer", 8, "soccer_usa_mls"),
        "LA_LIGA": _league_cfg("soccer", 9, "soccer_spain_la_liga"),
        "SERIE_A": _league_cfg("soccer", 10, "soccer_italy_serie_a"),
        "BUNDESLIGA": _league_cfg("soccer", 11, "soccer_germany_bundesliga"),
        "LIGUE_1": _league_cfg("soccer", 12, "soccer_france_ligue_one"),
        "UEFA_CL": _league_cfg("soccer", 14, "soccer_uefa_champs_league"),
    }
    for league in omit:
        leagues.pop(league, None)
    return {
        "leagues": leagues,
        "quota": {"min_remaining_requests": 25, "low_priority_min_remaining": 50},
        "defaults": {"max_events_per_league_per_run": 6, "max_leagues_per_run": 8},
    }


def _snapshots() -> pd.DataFrame:
    rows = []
    for league, sport, hours_ago in [
        ("NBA", "basketball", 2), ("WNBA", "basketball", 3),
        ("MLB", "baseball", 1), ("NHL", "hockey", 4),
    ]:
        rows.append({
            "league": league, "sport": sport,
            "snapshot_time": (NOW - pd.Timedelta(hours=hours_ago)).isoformat(),
            "prop_type": "points", "bookmaker": "draftkings",
        })
    return pd.DataFrame(rows)


def _last_run() -> dict:
    return {
        "league_statuses": {
            "NBA/odds_api": "collected", "WNBA/odds_api": "collected",
            "NCAAB/odds_api": "collected", "MLB/odds_api": "collected",
            "NHL/odds_api": "collected", "NFL/odds_api": "collected",
            "NCAAF/odds_api": "skipped_league_cap", "EPL/odds_api": "collected",
            "MLS/odds_api": "collected", "LA_LIGA/odds_api": "skipped_league_cap",
            "SERIE_A/odds_api": "skipped_quota_low",
            "BUNDESLIGA/odds_api": "skipped_quota_low_priority",
            "LIGUE_1/odds_api": "skipped_league_cap",
            "UEFA_CL/odds_api": "skipped_league_cap",
        },
    }


def _discovery() -> dict:
    active = {"NBA", "WNBA", "MLB", "NHL", "NFL", "NCAAF"}
    leagues = list(_config()["leagues"])
    return {
        "configured_sport_keys": [
            {"league": lg, "sport_key": "x", "available": True, "active": lg in active}
            for lg in leagues
        ],
    }


def _coverage(**kwargs):
    return build_coverage(
        kwargs.get("config", _config()),
        kwargs.get("snapshots", _snapshots()),
        kwargs.get("last_run", _last_run()),
        kwargs.get("discovery", _discovery()),
        kwargs.get("raw_file_counts", {"NBA": 12, "MLB": 30}),
        NOW,
    )


class SportGroupDetectionTests(unittest.TestCase):
    def test_all_five_sport_groups_present(self) -> None:
        summary = _coverage()
        groups = {g["sport_group"] for g in summary["sport_groups"]}
        self.assertEqual(groups, {"basketball", "baseball", "hockey", "football", "soccer"})

    def test_basketball_leagues_detected(self) -> None:
        summary = _coverage()
        rows = {r["league"]: r for r in summary["leagues"] if r["sport_group"] == "basketball"}
        self.assertEqual(set(rows), {"NBA", "WNBA", "NCAAB"})
        self.assertEqual(rows["NBA"]["status"], "collecting")
        self.assertEqual(rows["WNBA"]["status"], "collecting")
        self.assertTrue(rows["NBA"]["modeling_priority"])
        self.assertEqual(rows["NBA"]["priority"], 1)

    def test_baseball_league_detected(self) -> None:
        rows = {r["league"]: r for r in _coverage()["leagues"]}
        self.assertEqual(rows["MLB"]["sport_group"], "baseball")
        self.assertEqual(rows["MLB"]["status"], "collecting")
        self.assertEqual(rows["MLB"]["raw_files_saved"], 30)

    def test_hockey_league_detected(self) -> None:
        rows = {r["league"]: r for r in _coverage()["leagues"]}
        self.assertEqual(rows["NHL"]["sport_group"], "hockey")
        self.assertEqual(rows["NHL"]["status"], "collecting")

    def test_football_leagues_detected(self) -> None:
        rows = {r["league"]: r for r in _coverage()["leagues"] if r["sport_group"] == "football"}
        self.assertEqual(set(rows), {"NFL", "NCAAF"})

    def test_soccer_leagues_detected(self) -> None:
        rows = [r["league"] for r in _coverage()["leagues"] if r["sport_group"] == "soccer"]
        self.assertEqual(
            set(rows),
            {"EPL", "MLS", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGUE_1", "UEFA_CL"},
        )


class StatusClassificationTests(unittest.TestCase):
    def test_inactive_league_reported(self) -> None:
        rows = {r["league"]: r for r in _coverage()["leagues"]}
        # EPL collected nothing and the sport is inactive on the API.
        self.assertEqual(rows["EPL"]["status"], "configured_inactive")
        self.assertIn("off-season", rows["EPL"]["likely_reason"])

    def test_quota_skipped_league_reported(self) -> None:
        rows = {r["league"]: r for r in _coverage()["leagues"]}
        for league in ["LA_LIGA", "SERIE_A", "BUNDESLIGA", "NCAAF", "UEFA_CL"]:
            self.assertEqual(rows[league]["status"], "configured_skipped_quota", league)
            self.assertTrue(rows[league]["quota_blocked_last_run"], league)
        self.assertIn("LA_LIGA", _coverage()["leagues_by_status"]["configured_skipped_quota"])

    def test_active_league_with_no_events(self) -> None:
        rows = {r["league"]: r for r in _coverage()["leagues"]}
        self.assertEqual(rows["NFL"]["status"], "configured_no_events")
        self.assertIn("no prop-bearing events", rows["NFL"]["likely_reason"])

    def test_not_configured_league_reported(self) -> None:
        summary = _coverage(config=_config(omit={"NCAAF"}))
        rows = {r["league"]: r for r in summary["leagues"]}
        self.assertEqual(rows["NCAAF"]["status"], "not_configured")
        self.assertFalse(rows["NCAAF"]["configured"])
        self.assertTrue(any("NCAAF" in w and "missing from config" in w for w in summary["warnings"]))

    def test_collecting_beats_other_statuses(self) -> None:
        status, _ = league_status(
            configured=True, snapshots_last_24h=5,
            last_run_status="skipped_league_cap", sport_active=False,
        )
        self.assertEqual(status, "collecting")

    def test_error_status(self) -> None:
        status, reason = league_status(
            configured=True, snapshots_last_24h=0,
            last_run_status="error", sport_active=True,
        )
        self.assertEqual(status, "error")
        self.assertIn("run log", reason)

    def test_football_soccer_warnings_when_not_collecting(self) -> None:
        summary = _coverage()
        self.assertTrue(any(w.startswith("football:") for w in summary["warnings"]))
        self.assertTrue(any(w.startswith("soccer:") for w in summary["warnings"]))

    def test_markdown_renders(self) -> None:
        md = render_markdown(_coverage())
        self.assertIn("# Cross-Sport Collection Coverage", md)
        self.assertIn("Zero-snapshot leagues", md)
        self.assertIn("Research-only", md)


class DashboardSectionTests(unittest.TestCase):
    def test_dashboard_section_renders(self) -> None:
        from reports.dashboard import _build_cross_sport_coverage_section

        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder)
            (report_path / "cross_sport_collection_coverage_summary.json").write_text(
                json.dumps(_coverage(), default=str), encoding="utf-8"
            )
            section = _build_cross_sport_coverage_section(report_path)

        self.assertIn("Cross-Sport Coverage", section)
        self.assertIn("Sport groups covered", section)
        self.assertIn("NBA", section)
        self.assertIn("Quota-skipped", section)
        self.assertIn("soccer:", section)  # the not-collecting warning

    def test_dashboard_section_empty_state(self) -> None:
        from reports.dashboard import _build_cross_sport_coverage_section

        with tempfile.TemporaryDirectory() as folder:
            section = _build_cross_sport_coverage_section(Path(folder))
        self.assertIn("No coverage audit yet", section)


if __name__ == "__main__":
    unittest.main()
