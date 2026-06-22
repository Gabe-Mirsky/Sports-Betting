"""Offline tests for the SportsGameOdds historical player-prop probe.

No network, no real key: HTTP is injected, raw saves go to a temp directory.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import probe_sportsgameodds_historical_props as probe  # noqa: E402
from data.sportsgameodds_client import SportsGameOddsClient  # noqa: E402


def fake_http(responses):
    """Injectable http_get returning queued (status, headers, body) tuples."""

    calls = []

    def http_get(url, headers, timeout):
        calls.append({"url": url, "headers": dict(headers)})
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index]

    http_get.calls = calls
    return http_get


def usage_body(entities: int) -> str:
    return json.dumps({
        "success": True,
        "data": {
            "tier": "amateur",
            "rateLimits": {"per-month": {"max-entities": 2500, "current-entities": entities}},
        },
    })


def make_event(*, game_date: str = "2024-02-02", with_close_field: bool = True,
               with_results: bool = True) -> dict:
    prop_odd = {
        "oddID": "points-PLAYER_ONE_1_NBA-game-ou-over",
        "playerID": "PLAYER_ONE_1_NBA",
        "statEntityID": "PLAYER_ONE_1_NBA",
        "statID": "points",
        "openBookOdds": "-110",
        "openBookOverUnder": "25.5",
        "bookOdds": "-115",
        "bookOverUnder": "26.5",
        "score": 28,
        "started": True,
        "ended": True,
        "cancelled": False,
        "byBookmaker": {
            "draftkings": {
                "odds": "-115",
                "overUnder": "26.5",
                "openOdds": "-110",
                "openOverUnder": "25.5",
                "lastUpdatedAt": f"{game_date}T00:09:00Z",
            }
        },
    }
    if with_close_field:
        prop_odd["closeBookOdds"] = "-115"
    event = {
        "eventID": f"EVT_{game_date}",
        "leagueID": "NBA",
        "status": {"startsAt": f"{game_date}T00:10:00Z", "completed": True, "finalized": True},
        "teams": {
            "home": {"names": {"short": "NYK"}},
            "away": {"names": {"short": "SAS"}},
        },
        "odds": {
            prop_odd["oddID"]: prop_odd,
            "points-home-game-ml-home": {
                "oddID": "points-home-game-ml-home",
                "statEntityID": "home",
                "statID": "points",
                "bookOdds": "-200",
            },
        },
    }
    if with_results:
        event["results"] = {"game": {"home": 110, "away": 102}}
    return event


def events_body(event: dict) -> str:
    return json.dumps({"success": True, "data": [event]})


class AnalyzeEventTests(unittest.TestCase):
    def test_counts_and_field_detection(self) -> None:
        info = probe.analyze_event(make_event())
        self.assertEqual(info["n_odds"], 2)
        self.assertEqual(info["n_player_prop_odds"], 1)
        self.assertEqual(info["n_game_odds"], 1)
        self.assertEqual(info["player_props_with_open_price"], 1)
        self.assertEqual(info["player_props_with_close_field"], 1)
        self.assertEqual(info["player_props_with_final_book_price"], 1)
        self.assertEqual(info["player_props_with_score"], 1)
        self.assertTrue(info["has_results"])
        self.assertEqual(info["game_date"], "2024-02-02")
        self.assertEqual(info["home_team"], "NYK")
        self.assertEqual(info["away_team"], "SAS")
        self.assertEqual(info["bookmakers_seen"], ["draftkings"])
        self.assertIn("openOdds", info["per_book_open_fields"])
        self.assertIn("lastUpdatedAt", info["per_book_timestamp_fields"])
        self.assertIn("closeBookOdds", info["close_field_names"])
        self.assertIn("score", info["settlement_field_names"])
        self.assertIsNotNone(info["sample_player_prop"])

    def test_is_player_prop_odd(self) -> None:
        self.assertTrue(probe.is_player_prop_odd("x", {"playerID": "LEBRON_JAMES_1_NBA"}))
        self.assertFalse(probe.is_player_prop_odd(
            "points-home-game-ml-home", {"statEntityID": "home"}))
        # oddID fallback when the odds object carries no entity info.
        self.assertTrue(probe.is_player_prop_odd(
            "assists-DEVIN_VASSELL_1_NBA-game-ou-under", None))
        self.assertFalse(probe.is_player_prop_odd("points-away-game-ml-away", None))


class BuildWindowsTests(unittest.TestCase):
    def test_six_windows_newest_first(self) -> None:
        now = datetime(2026, 6, 11, tzinfo=timezone.utc)
        windows = probe.build_windows(now)
        self.assertEqual(len(windows), 6)
        self.assertEqual(windows[0]["name"], "this_week_completed")
        for window in windows:
            self.assertLess(window["starts_after"], window["starts_before"])
        anchors = [w["anchor_date"] for w in windows]
        self.assertEqual(anchors, sorted(anchors, reverse=True))


class BuildVerdictTests(unittest.TestCase):
    @staticmethod
    def window(name: str, *, ok: bool = True, current: bool = False,
               event: dict | None = None, status: int = 200,
               error: str | None = None) -> dict:
        return {
            "name": name, "ok": ok, "status": status, "error": error,
            "events_returned": 1 if event else 0,
            "is_current_week": current,
            "event": probe.analyze_event(event) if event else None,
        }

    def test_historical_props_with_explicit_close(self) -> None:
        windows = [
            self.window("now", current=True, event=make_event(game_date="2026-06-10")),
            self.window("old", event=make_event(game_date="2022-02-02")),
        ]
        verdict = probe.build_verdict(windows, "amateur", 8)
        self.assertTrue(verdict["historical_events_accessible"])
        self.assertTrue(verdict["historical_player_props_accessible"])
        self.assertTrue(verdict["open_close_prices_available"])
        self.assertTrue(verdict["closing_prices_available_for_props"])
        self.assertEqual(verdict["closing_price_form"], "explicit close* fields")
        self.assertTrue(verdict["settlement_results_available"])
        self.assertFalse(verdict["free_tier_blocked"])
        self.assertEqual(verdict["oldest_successful_game_date"], "2022-02-02")
        self.assertEqual(verdict["entity_cost_estimate"], 8)
        self.assertIn("backfill", verdict["recommended_next_step"])

    def test_final_book_price_counts_as_closing(self) -> None:
        windows = [self.window(
            "old", event=make_event(game_date="2023-01-15", with_close_field=False))]
        verdict = probe.build_verdict(windows, "amateur", 2)
        self.assertTrue(verdict["closing_prices_available_for_props"])
        self.assertIn("bookOdds", verdict["closing_price_form"])

    def test_current_week_only_is_not_historical(self) -> None:
        windows = [self.window("now", current=True, event=make_event())]
        verdict = probe.build_verdict(windows, "amateur", 1)
        self.assertFalse(verdict["historical_events_accessible"])
        self.assertFalse(verdict["historical_player_props_accessible"])

    def test_403_marks_free_tier_blocked(self) -> None:
        windows = [self.window("old", ok=False, status=403, error="http_403: forbidden")]
        verdict = probe.build_verdict(windows, "amateur", 0)
        self.assertTrue(verdict["free_tier_blocked"])
        self.assertFalse(verdict["historical_events_accessible"])


class RunProbeOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._raw_dir = probe.RAW_DIR
        self._tmp = tempfile.mkdtemp(prefix="sgo_hist_probe_test_")
        probe.RAW_DIR = Path(self._tmp)

    def tearDown(self) -> None:
        probe.RAW_DIR = self._raw_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_full_ladder_offline(self) -> None:
        responses = [(200, {}, usage_body(100))]
        for date in ("2026-06-10", "2026-01-12", "2025-06-08",
                     "2024-06-08", "2022-06-08", "2016-06-08"):
            responses.append((200, {}, events_body(make_event(game_date=date))))
        responses.append((200, {}, usage_body(108)))
        client = SportsGameOddsClient(api_key="k", http_get=fake_http(responses))
        summary = probe.run_probe(
            client,
            now=datetime(2026, 6, 11, tzinfo=timezone.utc),
            pause=0.0, max_windows=6, limit=1, sleep=lambda s: None,
        )
        self.assertEqual(summary["requests_made"], 8)
        self.assertEqual(summary["entity_cost"], 8)
        self.assertEqual(summary["tier"], "amateur")
        self.assertTrue(summary["historical_events_accessible"])
        self.assertTrue(summary["historical_player_props_accessible"])
        self.assertTrue(summary["closing_prices_available_for_props"])
        self.assertTrue(summary["settlement_results_available"])
        self.assertEqual(summary["oldest_successful_game_date"], "2016-06-08")
        self.assertEqual(summary["rejected_params"], [])
        self.assertIn("field_evidence", summary)
        self.assertIsNotNone(summary["sample_player_prop"])
        # Every request's raw payload was archived (8 files).
        self.assertEqual(len(list(Path(self._tmp).glob("*.json"))), 8)
        md = probe.render_md(summary)
        self.assertIn("historical_player_props_accessible: **True**", md)
        self.assertIn("backfill plan", md.lower())

    def test_http_400_drops_optional_params(self) -> None:
        responses = [
            (200, {}, usage_body(50)),
            (400, {}, json.dumps({"success": False, "error": "unknown param finalized"})),
            (200, {}, events_body(make_event(game_date="2026-06-10"))),
            (200, {}, usage_body(51)),
        ]
        http = fake_http(responses)
        client = SportsGameOddsClient(api_key="k", http_get=http)
        summary = probe.run_probe(
            client,
            now=datetime(2026, 6, 11, tzinfo=timezone.utc),
            pause=0.0, max_windows=1, limit=1, sleep=lambda s: None,
        )
        self.assertEqual(summary["rejected_params"], ["finalized"])
        self.assertNotIn("finalized=", http.calls[2]["url"])
        self.assertIn("includeOpenCloseOdds=", http.calls[2]["url"])
        self.assertTrue(summary["windows"][0]["ok"])

    def test_empty_history_recommends_no_backfill(self) -> None:
        responses = [
            (200, {}, usage_body(10)),
            (200, {}, events_body(make_event(game_date="2026-06-10"))),
            (200, {}, json.dumps({"success": True, "data": []})),
            (200, {}, usage_body(12)),
        ]
        client = SportsGameOddsClient(api_key="k", http_get=fake_http(responses))
        summary = probe.run_probe(
            client,
            now=datetime(2026, 6, 11, tzinfo=timezone.utc),
            pause=0.0, max_windows=2, limit=1, sleep=lambda s: None,
        )
        self.assertFalse(summary["historical_events_accessible"])
        self.assertFalse(summary["historical_player_props_accessible"])
        md = probe.render_md(summary)
        self.assertIn("## Blocker", md)


if __name__ == "__main__":
    unittest.main()
