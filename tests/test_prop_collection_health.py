from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.odds_api_sports import discover_available_sports  # noqa: E402
from data.prop_collection import (  # noqa: E402
    detect_collection_gap,
    leagues_in_priority_order,
    load_prop_collection_config,
    load_run_history,
    run_prop_collection,
)
from data.prop_collection_health import (  # noqa: E402
    build_health_summary,
    write_health_reports,
)
from reports.dashboard import write_static_dashboard_pages  # noqa: E402


NOW = datetime(2026, 6, 9, 17, 0, 0, tzinfo=timezone.utc)


def _league_cfg(sport_key: str, priority: int = 100, sport: str = "basketball", **extra) -> dict:
    cfg = {
        "sport": sport,
        "enabled": True,
        "modeling_priority": False,
        "collect_only": True,
        "priority": priority,
        "sources": {
            "odds_api": {
                "sport_key": sport_key,
                "markets": {"player_points": "points"},
            }
        },
    }
    cfg.update(extra)
    return cfg


def _config(leagues: dict | None = None, **overrides) -> dict:
    config = {
        "defaults": {
            "event_horizon_hours": 36,
            "closing_window_minutes": 60,
            "max_events_per_league_per_run": 6,
            "max_leagues_per_run": 0,
        },
        "quota": {"min_remaining_requests": 0},
        "catch_up": {"max_gap_hours": 24},
        "closing_snapshot": {"window_minutes": 60},
        "output": {
            "raw_dir": "data/raw/prop_odds",
            "processed_path": "data/processed/player_prop_snapshots_normalized.csv",
            "run_summary_path": "data/reports/player_prop_collection_run_summary.json",
            "run_log_dir": "data/logs/prop_collection_runs",
            "run_history_path": "data/reports/prop_collection_run_history.jsonl",
        },
        "sources": {"odds_api": {"enabled": True, "api_key_env": "ODDS_API_KEY"}},
        "leagues": leagues
        if leagues is not None
        else {"NBA": _league_cfg("basketball_nba", priority=1, modeling_priority=True, collect_only=False)},
    }
    config.update(overrides)
    return config


def _event(event_id: str = "ev1", minutes_out: float = 30.0, sport_key: str = "basketball_nba") -> dict:
    commence = (NOW + timedelta(minutes=minutes_out)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": event_id,
        "sport_key": sport_key,
        "commence_time": commence,
        "home_team": "Oklahoma City Thunder",
        "away_team": "Houston Rockets",
    }


def _event_odds(event_id: str = "ev1", minutes_out: float = 30.0) -> dict:
    payload = _event(event_id, minutes_out)
    payload["bookmakers"] = [
        {
            "key": "draftkings",
            "last_update": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "markets": [
                {
                    "key": "player_points",
                    "last_update": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "outcomes": [
                        {"name": "Over", "description": "Shai Gilgeous-Alexander", "price": 1.91, "point": 31.5},
                        {"name": "Under", "description": "Shai Gilgeous-Alexander", "price": 1.91, "point": 31.5},
                    ],
                }
            ],
        }
    ]
    return payload


def _fake_fetch(events: list[dict], odds_by_event: dict[str, dict], quota_remaining: float | None = None):
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        if "/events?" in url or url.endswith("/events"):
            return events
        for event_id, payload in odds_by_event.items():
            if f"/events/{event_id}/odds" in url:
                return payload
        raise AssertionError(f"unexpected url: {url}")

    fetch.calls = calls  # type: ignore[attr-defined]
    fetch.quota_remaining = quota_remaining  # type: ignore[attr-defined]
    return fetch


def _run(tmp: Path, config: dict, fetch, env: dict | None = None, now: datetime = NOW) -> dict:
    return run_prop_collection(
        config, tmp, now=now, fetch_json=fetch,
        env=env if env is not None else {"ODDS_API_KEY": "test-key"},
    )


class HealthSummaryTests(unittest.TestCase):
    def test_health_summary_after_successful_runs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            _run(
                tmp, config,
                _fake_fetch([_event("ev2", minutes_out=120.0)], {"ev2": _event_odds("ev2", minutes_out=120.0)}),
                now=NOW + timedelta(hours=4),
            )

            summary = build_health_summary(
                config, tmp, now=NOW + timedelta(hours=5), env={"ODDS_API_KEY": "test-key"}
            )
            self.assertTrue(summary["healthy"], summary["health_reasons"])
            self.assertEqual(summary["runs"]["total"], 2)
            self.assertEqual(summary["runs"]["successful"], 2)
            self.assertEqual(summary["runs"]["failed"], 0)
            self.assertEqual(len(summary["snapshots_by_run"]), 2)
            self.assertEqual(summary["snapshots_by_sport"].get("basketball"), 2)
            self.assertTrue(summary["api_key_detected"])
            self.assertFalse(summary["likely_quota_issue"])
            self.assertIsNotNone(summary["last_successful_collection_utc"])
            self.assertIsNone(summary["last_failed_collection_utc"])
            self.assertEqual(summary["missed_days_count"], 0)
            self.assertTrue(summary["latest_run_log"])

    def test_health_reports_written_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            summary = write_health_reports(config, tmp, now=NOW + timedelta(hours=1), env={"ODDS_API_KEY": "k"})
            json_path = tmp / "data" / "reports" / "prop_collection_health_summary.json"
            md_path = tmp / "data" / "reports" / "prop_collection_health.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["report"], "prop_collection_health")
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("Prop Collection Health", md)
            self.assertIn("HEALTHY", md)
            self.assertIn("research-only", md)
            self.assertTrue(summary["research_only"])

    def test_unhealthy_without_api_key_and_stale_runs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            summary = build_health_summary(config, tmp, now=NOW + timedelta(days=3), env={})
            self.assertFalse(summary["healthy"])
            reasons = " ".join(summary["health_reasons"])
            self.assertIn("ODDS_API_KEY", reasons)
            self.assertIn("Last successful collection", reasons)
            # Two full calendar days (Jun 10, 11) had no runs at all.
            self.assertGreaterEqual(summary["missed_days_count"], 2)

    def test_health_backfills_runs_from_logs_without_history_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            log_dir = tmp / "data" / "logs" / "prop_collection_runs"
            log_dir.mkdir(parents=True)
            run_id = "20260608T120000Z"
            (log_dir / f"run_{run_id}.log").write_text(
                f"2026-06-08T12:00:00+00:00 run {run_id}: starting multi-sport prop collection (research-only)\n"
                "2026-06-08T12:00:01+00:00 NBA/odds_api: collected (10 snapshots from 1 events)\n"
                "2026-06-08T12:00:02+00:00 WNBA/odds_api: skipped (ODDS_API_KEY not set)\n"
                f"2026-06-08T12:00:03+00:00 run {run_id}: done — 10 added, 0 duplicates removed, 10 total snapshots\n",
                encoding="utf-8",
            )
            summary = build_health_summary(config, tmp, now=NOW, env={"ODDS_API_KEY": "k"})
            self.assertEqual(summary["runs"]["total"], 1)
            self.assertEqual(summary["runs"]["successful"], 1)
            run = summary["snapshots_by_run"][0]
            self.assertEqual(run["run_id"], run_id)
            self.assertEqual(run["snapshots_collected"], 10)


class MissedRunDetectionTests(unittest.TestCase):
    def test_gap_warning_written_when_last_run_too_old(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            late = NOW + timedelta(hours=30)
            summary = _run(
                tmp, config,
                _fake_fetch([_event("ev2", minutes_out=60.0)], {"ev2": _event_odds("ev2", minutes_out=60.0)}),
                now=late,
            )
            self.assertEqual(len(summary["warnings"]), 1)
            warning = summary["warnings"][0]
            self.assertEqual(warning["type"], "missed_collection_window")
            self.assertGreater(warning["gap_hours"], 24)
            # Honest about unrecoverable data.
            self.assertIn("NOT recoverable", warning["message"])
            # Collection still happened normally.
            self.assertEqual(summary["leagues"][0]["status"], "collected")
            log_text = (tmp / summary["outputs"]["run_log"]).read_text(encoding="utf-8")
            self.assertIn("WARNING", log_text)

    def test_no_gap_warning_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            summary = _run(
                tmp, config,
                _fake_fetch([_event("ev2", minutes_out=60.0)], {"ev2": _event_odds("ev2", minutes_out=60.0)}),
                now=NOW + timedelta(hours=12),
            )
            self.assertEqual(summary["warnings"], [])

    def test_detect_collection_gap_pure(self) -> None:
        history = [{"run_time_utc": (NOW - timedelta(hours=40)).isoformat()}]
        warning = detect_collection_gap(history, NOW, 24)
        self.assertIsNotNone(warning)
        self.assertAlmostEqual(warning["gap_hours"], 40.0, places=1)
        self.assertIsNone(detect_collection_gap(history, NOW, 48))
        self.assertIsNone(detect_collection_gap([], NOW, 24))

    def test_run_history_appends_one_record_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}), now=NOW + timedelta(hours=1))
            history = load_run_history(tmp / "data" / "reports" / "prop_collection_run_history.jsonl")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["outcome"], "success")
            self.assertEqual(history[0]["snapshots_by_league"], {"NBA": 1})


class StartupBatchTests(unittest.TestCase):
    def test_startup_batch_runs_collection_settlement_health_and_dashboard(self) -> None:
        batch = (PROJECT_ROOT / "run_prop_collection_startup.bat").read_text(encoding="utf-8")
        self.assertIn('cd /d "%~dp0"', batch)
        self.assertIn("scripts\\daily_collect_props.py", batch)
        self.assertIn("scripts\\refresh_nba_results_and_settle_props.py", batch)
        self.assertIn("scripts\\build_prop_collection_health.py", batch)
        self.assertIn("scripts\\build_dashboard.py", batch)
        # Every step logs to the same startup log, including exit codes.
        self.assertIn('>> "%LOG%" 2>&1', batch)
        self.assertIn("exit code", batch)
        self.assertIn("startup_runs", batch)


class SoccerConfigTests(unittest.TestCase):
    def test_repo_config_adds_soccer_leagues_collect_only(self) -> None:
        config = load_prop_collection_config(PROJECT_ROOT / "config" / "prop_collection.yaml")
        leagues = config["leagues"]
        expected_keys = {
            "EPL": "soccer_epl",
            "MLS": "soccer_usa_mls",
            "LA_LIGA": "soccer_spain_la_liga",
            "SERIE_A": "soccer_italy_serie_a",
            "BUNDESLIGA": "soccer_germany_bundesliga",
            "LIGUE_1": "soccer_france_ligue_one",
        }
        for league, sport_key in expected_keys.items():
            self.assertIn(league, leagues, league)
            cfg = leagues[league]
            self.assertEqual(cfg["sport"], "soccer")
            self.assertTrue(cfg["enabled"], league)
            self.assertTrue(cfg["collect_only"], league)
            self.assertFalse(cfg["modeling_priority"], league)
            self.assertEqual(cfg["sources"]["odds_api"]["sport_key"], sport_key)
            self.assertTrue(cfg["sources"]["odds_api"]["markets"])
        # NBA stays the only modeling-priority league and outranks all soccer.
        order = [league for league, _ in leagues_in_priority_order(config)]
        self.assertEqual(order[0], "NBA")
        for league in expected_keys:
            self.assertGreater(order.index(league), order.index("NBA"))

    def test_disabled_soccer_league_is_skipped_without_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            leagues = {
                "EPL": _league_cfg("soccer_epl", priority=7, sport="soccer", enabled=False),
            }

            def explode(url: str):
                raise AssertionError("fetch must not be called for disabled leagues")

            summary = _run(tmp, _config(leagues), explode)
            self.assertEqual(summary["leagues"][0]["status"], "skipped_disabled")
            self.assertEqual(summary["totals"]["snapshots_total"], 0)


class SportsDiscoveryTests(unittest.TestCase):
    def test_discovery_writes_report_and_flags_configured_keys(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            leagues = {
                "NBA": _league_cfg("basketball_nba", priority=1),
                "EPL": _league_cfg("soccer_epl", priority=7, sport="soccer"),
                "BAD": _league_cfg("not_a_real_sport", priority=99),
            }
            config = _config(leagues)
            sports = [
                {"key": "basketball_nba", "group": "Basketball", "title": "NBA", "active": True},
                {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": False},
                {"key": "soccer_usa_mls", "group": "Soccer", "title": "MLS", "active": True},
            ]

            def fetch(url: str):
                assert "/sports?" in url
                return sports

            summary = discover_available_sports(config, tmp, "test-key", fetch_json=fetch, now=NOW)
            report_path = tmp / "data" / "reports" / "odds_api_available_sports.json"
            self.assertTrue(report_path.exists())
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["sports_count"], 3)
            self.assertEqual(loaded["soccer_count"], 2)
            status = {entry["league"]: entry for entry in summary["configured_sport_keys"]}
            self.assertTrue(status["NBA"]["available"])
            self.assertTrue(status["NBA"]["active"])
            self.assertTrue(status["EPL"]["available"])
            self.assertFalse(status["EPL"]["active"])
            self.assertFalse(status["BAD"]["available"])


class QuotaGuardTests(unittest.TestCase):
    def test_low_quota_skips_lower_priority_leagues(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            leagues = {
                "NBA": _league_cfg("basketball_nba", priority=1),
                "EPL": _league_cfg("soccer_epl", priority=7, sport="soccer"),
            }
            config = _config(leagues, quota={"min_remaining_requests": 50})
            # The fake reports 10 credits remaining before any league runs,
            # so the guard skips every league instead of burning credits.
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()}, quota_remaining=10.0)
            summary = _run(tmp, config, fetch)
            statuses = {r["league"]: r["status"] for r in summary["leagues"]}
            self.assertEqual(statuses["NBA"], "skipped_quota_low")
            self.assertEqual(statuses["EPL"], "skipped_quota_low")
            self.assertTrue(summary["quota"]["likely_quota_issue"])

    def test_quota_unknown_does_not_trigger_guard(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config(quota={"min_remaining_requests": 50})
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()}, quota_remaining=None)
            summary = _run(tmp, config, fetch)
            self.assertEqual(summary["leagues"][0]["status"], "collected")
            self.assertFalse(summary["quota"]["likely_quota_issue"])

    def test_max_leagues_per_run_caps_in_priority_order(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            leagues = {
                # Config order intentionally differs from priority order.
                "EPL": _league_cfg("soccer_epl", priority=7, sport="soccer"),
                "NBA": _league_cfg("basketball_nba", priority=1),
            }
            config = _config(leagues)
            config["defaults"]["max_leagues_per_run"] = 1
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            summary = _run(tmp, config, fetch)
            statuses = {r["league"]: r["status"] for r in summary["leagues"]}
            self.assertEqual(statuses["NBA"], "collected")
            self.assertEqual(statuses["EPL"], "skipped_league_cap")

    def test_per_league_max_events_override(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            leagues = {"NBA": _league_cfg("basketball_nba", priority=1, max_events_per_run=1)}
            events = [_event("ev1"), _event("ev2", minutes_out=90.0)]
            odds = {"ev1": _event_odds("ev1"), "ev2": _event_odds("ev2", minutes_out=90.0)}
            fetch = _fake_fetch(events, odds)
            summary = _run(tmp, _config(leagues), fetch)
            self.assertEqual(summary["leagues"][0]["events"], 1)


class DashboardHealthSectionTests(unittest.TestCase):
    def test_player_props_page_shows_health_section(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config()
            _run(tmp, config, _fake_fetch([_event()], {"ev1": _event_odds()}))
            write_health_reports(config, tmp, now=NOW + timedelta(hours=1), env={"ODDS_API_KEY": "k"})

            reports_dir = tmp / "data" / "reports"
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            html = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", html)
            self.assertIn("Advanced reports", html)
            self.assertIn("prop_collection_health.md", html)
            self.assertIn("Games Collected", html)
            self.assertNotIn("Snapshots By Sport (health report)", html)
            self.assertNotIn("Latest Errors / Warnings", html)


if __name__ == "__main__":
    unittest.main()
