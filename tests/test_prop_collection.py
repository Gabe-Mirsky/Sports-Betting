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

from data.prop_collection import (  # noqa: E402
    enabled_leagues,
    load_prop_collection_config,
    run_prop_collection,
)
from reports.dashboard import write_static_dashboard_pages  # noqa: E402


NOW = datetime(2026, 6, 9, 17, 0, 0, tzinfo=timezone.utc)


def _config(tmp: Path, leagues: dict | None = None) -> dict:
    return {
        "defaults": {
            "event_horizon_hours": 36,
            "closing_window_minutes": 60,
            "max_events_per_league_per_run": 6,
        },
        "closing_snapshot": {"window_minutes": 60},
        "output": {
            "raw_dir": "data/raw/prop_odds",
            "processed_path": "data/processed/player_prop_snapshots_normalized.csv",
            "run_summary_path": "data/reports/player_prop_collection_run_summary.json",
            "run_log_dir": "data/logs/prop_collection_runs",
        },
        "sources": {"odds_api": {"enabled": True, "api_key_env": "ODDS_API_KEY"}},
        "leagues": leagues
        if leagues is not None
        else {
            "NBA": {
                "sport": "basketball",
                "enabled": True,
                "modeling_priority": True,
                "collect_only": False,
                "sources": {
                    "odds_api": {
                        "sport_key": "basketball_nba",
                        "markets": {"player_points": "points"},
                    }
                },
            }
        },
    }


def _event(event_id: str = "ev1", minutes_out: float = 30.0) -> dict:
    commence = (NOW + timedelta(minutes=minutes_out)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": event_id,
        "sport_key": "basketball_nba",
        "commence_time": commence,
        "home_team": "Oklahoma City Thunder",
        "away_team": "Houston Rockets",
    }


def _event_odds(event_id: str = "ev1", minutes_out: float = 30.0, last_update: str | None = None) -> dict:
    payload = _event(event_id, minutes_out)
    payload["bookmakers"] = [
        {
            "key": "draftkings",
            "last_update": last_update or NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "markets": [
                {
                    "key": "player_points",
                    "last_update": last_update or NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "outcomes": [
                        {"name": "Over", "description": "Shai Gilgeous-Alexander", "price": 1.91, "point": 31.5},
                        {"name": "Under", "description": "Shai Gilgeous-Alexander", "price": 1.91, "point": 31.5},
                    ],
                }
            ],
        }
    ]
    return payload


def _fake_fetch(events: list[dict], odds_by_event: dict[str, dict]):
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
    return fetch


def _run(tmp: Path, config: dict, fetch, env: dict | None = None, now: datetime = NOW) -> dict:
    return run_prop_collection(config, tmp, now=now, fetch_json=fetch, env=env if env is not None else {"ODDS_API_KEY": "test-key"})


class ConfigTests(unittest.TestCase):
    def test_load_config_parses_leagues_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prop_collection.yaml"
            path.write_text(
                "leagues:\n"
                "  NBA:\n"
                "    sport: basketball\n"
                "    enabled: true\n"
                "    modeling_priority: true\n"
                "  WNBA:\n"
                "    sport: basketball\n"
                "    enabled: false\n",
                encoding="utf-8",
            )
            config = load_prop_collection_config(path)
            self.assertEqual(config["defaults"]["closing_window_minutes"], 60)
            self.assertEqual(config["closing_snapshot"]["window_minutes"], 60)
            self.assertIn("NBA", config["leagues"])
            self.assertEqual(list(enabled_leagues(config)), ["NBA"])

    def test_repo_config_marks_nba_modeling_priority_and_others_collect_only(self) -> None:
        config = load_prop_collection_config(PROJECT_ROOT / "config" / "prop_collection.yaml")
        leagues = config["leagues"]
        for league in ["NBA", "WNBA", "NFL", "MLB", "NHL", "NCAAB"]:
            self.assertIn(league, leagues)
        self.assertTrue(leagues["NBA"]["modeling_priority"])
        self.assertFalse(leagues["NBA"]["collect_only"])
        for league in ["WNBA", "NFL", "MLB", "NHL", "NCAAB"]:
            self.assertFalse(leagues[league]["modeling_priority"], league)
            self.assertTrue(leagues[league]["collect_only"], league)


class CollectionTests(unittest.TestCase):
    def test_disabled_sport_is_skipped_without_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config(tmp)
            config["leagues"]["NBA"]["enabled"] = False

            def explode(url: str):
                raise AssertionError("fetch must not be called for disabled leagues")

            summary = _run(tmp, config, explode)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["leagues"][0]["status"], "skipped_disabled")
            self.assertEqual(summary["totals"]["snapshots_total"], 0)

    def test_collect_only_sport_still_collects(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config(tmp)
            config["leagues"]["NBA"]["collect_only"] = True
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            summary = _run(tmp, config, fetch)
            record = summary["leagues"][0]
            self.assertEqual(record["status"], "collected")
            self.assertTrue(record["collect_only"])
            self.assertEqual(record["snapshots_collected"], 1)

    def test_raw_response_saved_with_timestamped_filename(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            summary = _run(tmp, _config(tmp), fetch)
            raw_dir = tmp / "data" / "raw" / "prop_odds" / "NBA" / "odds_api"
            files = sorted(p.name for p in raw_dir.glob("*.json"))
            run_id = summary["run_id"]
            self.assertIn(f"{run_id}__events.json", files)
            self.assertIn(f"{run_id}__ev1.json", files)
            payload = json.loads((raw_dir / f"{run_id}__ev1.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["id"], "ev1")

    def test_snapshots_append_without_overwriting_history(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config(tmp)
            first = _fake_fetch([_event()], {"ev1": _event_odds(last_update="2026-06-09T16:00:00Z")})
            _run(tmp, config, first)
            second = _fake_fetch([_event()], {"ev1": _event_odds(last_update="2026-06-09T16:45:00Z")})
            summary = _run(tmp, config, second, now=NOW + timedelta(minutes=45))
            csv_path = tmp / "data" / "processed" / "player_prop_snapshots_normalized.csv"
            frame = pd.read_csv(csv_path)
            self.assertEqual(len(frame), 2)
            times = set(pd.to_datetime(frame["snapshot_time"], utc=True).astype(str))
            self.assertEqual(len(times), 2)
            self.assertEqual(summary["totals"]["snapshots_added"], 1)

    def test_exact_duplicate_snapshots_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = _config(tmp)
            fetch = _fake_fetch([_event()], {"ev1": _event_odds(last_update="2026-06-09T16:00:00Z")})
            _run(tmp, config, fetch)
            summary = _run(tmp, config, fetch, now=NOW + timedelta(minutes=5))
            csv_path = tmp / "data" / "processed" / "player_prop_snapshots_normalized.csv"
            frame = pd.read_csv(csv_path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(summary["totals"]["snapshots_added"], 0)
            self.assertEqual(summary["totals"]["duplicates_removed"], 1)

    def test_closing_snapshot_flagged_inside_window_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            events = [_event("close", minutes_out=30.0), _event("far", minutes_out=300.0)]
            odds = {
                "close": _event_odds("close", minutes_out=30.0),
                "far": _event_odds("far", minutes_out=300.0),
            }
            fetch = _fake_fetch(events, odds)
            _run(tmp, _config(tmp), fetch)
            frame = pd.read_csv(tmp / "data" / "processed" / "player_prop_snapshots_normalized.csv")
            by_event = {row["market_id"].split(":")[0]: row for _, row in frame.iterrows()}
            self.assertTrue(bool(by_event["close"]["is_closing_snapshot"]))
            self.assertFalse(bool(by_event["far"]["is_closing_snapshot"]))
            self.assertAlmostEqual(float(by_event["close"]["minutes_to_game_start"]), 30.0, places=1)
            # Both early and closing snapshots are kept.
            self.assertEqual(len(frame), 2)

    def test_missing_odds_api_key_is_handled_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)

            def explode(url: str):
                raise AssertionError("fetch must not be called without an API key")

            summary = _run(tmp, _config(tmp), explode, env={})
            self.assertEqual(summary["status"], "ok")
            record = summary["leagues"][0]
            self.assertEqual(record["status"], "skipped_no_api_key")
            self.assertIn("ODDS_API_KEY", record["detail"])
            # Outputs still exist so downstream pages never break.
            self.assertTrue((tmp / "data" / "processed" / "player_prop_snapshots_normalized.csv").exists())
            self.assertTrue((tmp / "data" / "reports" / "player_prop_collection_run_summary.json").exists())

    def test_run_log_written(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            summary = _run(tmp, _config(tmp), fetch)
            log_path = tmp / summary["outputs"]["run_log"]
            self.assertTrue(log_path.exists())
            self.assertIn("research-only", log_path.read_text(encoding="utf-8"))


class NbaClosingPriorityTests(unittest.TestCase):
    """Phase: NBA near-tip prioritization + low-priority quota protection."""

    def _two_league_config(self, tmp: Path, nba_priority: int = 9, mlb_priority: int = 1) -> dict:
        # NBA deliberately gets a WORSE numeric priority than MLB to prove
        # modeling_priority wins over the numeric ordering.
        leagues = {
            "MLB": {
                "sport": "baseball",
                "enabled": True,
                "modeling_priority": False,
                "collect_only": True,
                "priority": mlb_priority,
                "sources": {
                    "odds_api": {
                        "sport_key": "baseball_mlb",
                        "markets": {"batter_hits": "hits"},
                    }
                },
            },
            "NBA": {
                "sport": "basketball",
                "enabled": True,
                "modeling_priority": True,
                "collect_only": False,
                "priority": nba_priority,
                "sources": {
                    "odds_api": {
                        "sport_key": "basketball_nba",
                        "markets": {"player_points": "points"},
                    }
                },
            },
        }
        return _config(tmp, leagues)

    def test_modeling_priority_league_collects_first(self) -> None:
        from data.prop_collection import leagues_in_priority_order

        with tempfile.TemporaryDirectory() as folder:
            config = self._two_league_config(Path(folder))
            ordered = [league for league, _ in leagues_in_priority_order(config)]
            self.assertEqual(ordered[0], "NBA")

    def test_low_priority_league_skipped_when_quota_limited(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = self._two_league_config(tmp)
            config["quota"] = {"min_remaining_requests": 25, "low_priority_min_remaining": 60}

            events = [_event()]
            odds = {"ev1": _event_odds()}
            inner = _fake_fetch(events, odds)

            def fetch(url: str):
                result = inner(url)
                fetch.quota_remaining = 40.0  # above NBA floor, below low-priority floor
                return result

            fetch.quota_remaining = 40.0
            summary = _run(tmp, config, fetch)

            statuses = {r["league"]: r["status"] for r in summary["leagues"]}
            self.assertEqual(statuses["NBA"], "collected")
            self.assertEqual(statuses["MLB"], "skipped_quota_low_priority")
            self.assertTrue(summary["quota"]["likely_quota_issue"])
            skipped = {entry["league"]: entry for entry in summary["leagues_skipped"]}
            self.assertIn("MLB", skipped)
            self.assertIn("protect NBA", skipped["MLB"]["reason"])

    def test_summary_flags_nba_near_tip_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            config = self._two_league_config(tmp)

            # Seed the snapshot store with an NBA game tipping in 45 minutes.
            processed = tmp / "data" / "processed"
            processed.mkdir(parents=True)
            start = NOW + timedelta(minutes=45)
            pd.DataFrame(
                [
                    {
                        "snapshot_time": (NOW - timedelta(hours=4)).isoformat(),
                        "league": "NBA",
                        "sport": "basketball",
                        "canonical_game_key": "basketball|NBA|2026-06-09|OKC|HOU",
                        "game_start_time": start.isoformat(),
                        "player_name": "Test Player",
                        "prop_type": "points",
                        "line": 25.5,
                        "bookmaker": "draftkings",
                        "is_closing_snapshot": False,
                    }
                ]
            ).to_csv(processed / "player_prop_snapshots_normalized.csv", index=False)

            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            summary = _run(tmp, config, fetch)

            closing = summary["nba_closing_priority"]
            self.assertTrue(closing["nba_within_2h"])
            self.assertTrue(closing["nba_within_60m"])
            self.assertTrue(closing["high_priority_run"])
            self.assertTrue(closing["nba_prioritized"])
            self.assertAlmostEqual(closing["minutes_until_next_nba_game"], 45.0, places=1)
            self.assertIn("NBA", summary["leagues_collected"])

    def test_no_nba_games_known_keeps_flags_off(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            summary = _run(tmp, _config(tmp), fetch)
            closing = summary["nba_closing_priority"]
            self.assertIsNone(closing["minutes_until_next_nba_game"])
            self.assertFalse(closing["nba_within_2h"])
            self.assertFalse(closing["high_priority_run"])


class DashboardPageTests(unittest.TestCase):
    def test_player_props_page_shows_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            fetch = _fake_fetch([_event()], {"ev1": _event_odds()})
            _run(tmp, _config(tmp), fetch)

            reports_dir = tmp / "data" / "reports"
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            html = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", html)
            self.assertIn("Total games", html)
            self.assertIn("Games Collected", html)
            self.assertIn("Complete Markets", html)
            self.assertIn("Research Bets We Would Place", html)
            self.assertIn("NBA", html)
            self.assertNotIn("Snapshots By Sport", html)
            self.assertNotIn("Missing player_id By Sport", html)


if __name__ == "__main__":
    unittest.main()
