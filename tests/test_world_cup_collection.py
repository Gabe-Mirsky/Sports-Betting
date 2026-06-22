"""Tests for World Cup collection: normalization, dry-run, no-key, quota guard.

No network: a fake fetcher returns canned Odds API payloads and drives the
quota-remaining attribute the guard reads.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.world_cup_collection import (  # noqa: E402
    WORLD_CUP_SNAPSHOT_COLUMNS, normalize_world_cup_odds, run_world_cup_collection,
)

NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


def _config(root: Path) -> dict:
    return {
        "source": {"enabled": True, "api_key_env": "ODDS_API_KEY",
                   "base_url": "https://api.the-odds-api.com/v4", "regions": "us",
                   "odds_format": "decimal"},
        "defaults": {"closing_window_minutes": 60.0, "max_events_per_run": 1},
        "quota": {"min_remaining_requests": 100, "max_credits_per_run": 2},
        "markets": {"h2h": "match_winner_1x2", "totals": "total_goals"},
        "leagues": {"WORLD_CUP": {"sport_key": "soccer_fifa_world_cup"}},
        "output": {
            "raw_dir": "data/raw/world_cup",
            "processed_path": "data/processed/world_cup_odds_snapshots_normalized.csv",
            "run_summary_path": "data/reports/world_cup_collection_summary.json",
            "run_history_path": "data/reports/world_cup_run_history.jsonl",
        },
    }


def _event(eid: str, minutes_from_now: int) -> dict:
    start = (NOW + timedelta(minutes=minutes_from_now)).isoformat().replace("+00:00", "Z")
    return {
        "id": eid, "sport_key": "soccer_fifa_world_cup", "commence_time": start,
        "home_team": "Argentina", "away_team": "Brazil",
        "bookmakers": [{
            "key": "draftkings", "title": "DraftKings",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Argentina", "price": 2.1},
                    {"name": "Brazil", "price": 3.4},
                    {"name": "Draw", "price": 3.1}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.9, "point": 2.5},
                    {"name": "Under", "price": 1.9, "point": 2.5}]},
            ],
        }],
    }


class FakeFetch:
    def __init__(self, events, odds, remaining_after_events=500.0):
        self.events, self.odds = events, odds
        self.remaining_after_events = remaining_after_events
        self.quota_remaining = None
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        if "/events" in url:
            self.quota_remaining = self.remaining_after_events
            return self.events
        if "/odds" in url:
            if self.remaining_after_events is not None:
                self.quota_remaining = self.remaining_after_events - 2
            return self.odds
        return []

    def odds_called(self) -> bool:
        return any("/odds" in u for u in self.calls)


class TestNormalize(unittest.TestCase):
    def test_columns_and_rows(self):
        frame = normalize_world_cup_odds([_event("e1", 30)], _config(Path(".")), run_time=NOW)
        self.assertEqual(list(frame.columns), list(WORLD_CUP_SNAPSHOT_COLUMNS))
        self.assertEqual(len(frame), 5)  # 3 h2h + 2 totals, one book
        self.assertEqual(set(frame["market_type"]), {"match_winner_1x2", "total_goals"})
        self.assertTrue(frame["is_closing_like"].all())  # 30 min out
        totals = frame[frame["market_type"] == "total_goals"]
        self.assertTrue((totals["line"] == 2.5).all())

    def test_not_closing_when_far_out(self):
        frame = normalize_world_cup_odds([_event("e1", 600)], _config(Path(".")), run_time=NOW)
        self.assertFalse(frame["is_closing_like"].any())

    def test_max_events_cap(self):
        payload = [_event("e1", 30), _event("e2", 40)]
        frame = normalize_world_cup_odds(payload, _config(Path(".")), run_time=NOW, max_events=1)
        self.assertEqual(frame["event_id"].nunique(), 1)


class TestRunCollection(unittest.TestCase):
    def test_no_key_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            s = run_world_cup_collection(_config(Path(d)), d, env={}, now=NOW)
        self.assertEqual(s["status"], "no_key")

    def test_dry_run_lists_events_only(self):
        with tempfile.TemporaryDirectory() as d:
            fetch = FakeFetch([_event("e1", 30)], [_event("e1", 30)])
            s = run_world_cup_collection(_config(Path(d)), d, dry_run=True,
                                         env={"ODDS_API_KEY": "k"}, fetch_json=fetch, now=NOW)
        self.assertEqual(s["status"], "dry_run_ok")
        self.assertEqual(s["events_found"], 1)
        self.assertFalse(fetch.odds_called())  # no credit-spending call

    def test_quota_guard_skips_odds(self):
        with tempfile.TemporaryDirectory() as d:
            fetch = FakeFetch([_event("e1", 30)], [_event("e1", 30)], remaining_after_events=50.0)
            s = run_world_cup_collection(_config(Path(d)), d, env={"ODDS_API_KEY": "k"},
                                         fetch_json=fetch, now=NOW)
        self.assertEqual(s["status"], "skipped_quota")
        self.assertFalse(fetch.odds_called())

    def test_collected_writes_normalized_csv(self):
        with tempfile.TemporaryDirectory() as d:
            fetch = FakeFetch([_event("e1", 30), _event("e2", 40)],
                              [_event("e1", 30), _event("e2", 40)])
            s = run_world_cup_collection(_config(Path(d)), d, max_events=1,
                                         env={"ODDS_API_KEY": "k"}, fetch_json=fetch, now=NOW)
            self.assertEqual(s["status"], "collected")
            self.assertEqual(s["events_processed"], 1)   # capped to 1
            self.assertEqual(s["rows_normalized"], 5)
            self.assertTrue(s["game_markets_found"])
            self.assertFalse(s["player_props_found"])
            self.assertTrue(fetch.odds_called())
            csv = Path(d) / "data/processed/world_cup_odds_snapshots_normalized.csv"
            self.assertTrue(csv.exists())


if __name__ == "__main__":
    unittest.main()
