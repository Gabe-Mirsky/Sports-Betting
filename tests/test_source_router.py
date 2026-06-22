"""Tests for the multi-source data router decision logic (pure; no network)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.source_router import (  # noqa: E402
    SKIP_NO_SAFE_SOURCE, SKIP_PAID_ODDS, SourceRouter, SourceState,
    best_source_by_data_type, load_router_config,
)

CONFIG = load_router_config(PROJECT_ROOT / "config" / "source_priority.yaml")
NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


def _states(**over) -> dict[str, SourceState]:
    base = {
        "odds_api": SourceState("odds_api", key_present=True, quota_remaining=500.0),
        "sportsgameodds": SourceState("sportsgameodds", key_present=True, quota_remaining=2000.0),
        "apisports": SourceState("apisports", key_present=True, quota_remaining=90.0),
        "kalshi": SourceState("kalshi", key_present=False),  # public, no key needed
    }
    base.update(over)
    return base


def _router(states, now=NOW):
    return SourceRouter(CONFIG, states, now=now)


class TestRouting(unittest.TestCase):
    def test_wc_schedule_prefers_apisports(self):
        d = _router(_states()).route("WORLD_CUP", "events_schedule")
        self.assertEqual(d.selected, "apisports")

    def test_wc_schedule_falls_back_to_free_odds_api_when_apisports_blocked(self):
        st = _states(apisports=SourceState("apisports", key_present=True, blocked_reason="plan_season_2026"))
        # odds_api quota is far below floor, but events_schedule is FREE for odds_api.
        st["odds_api"] = SourceState("odds_api", key_present=True, quota_remaining=10.0)
        d = _router(st).route("WORLD_CUP", "events_schedule")
        self.assertEqual(d.selected, "odds_api")
        self.assertEqual(d.skipped[0]["source"], "apisports")
        self.assertIn("blocked", d.skipped[0]["reason"])

    def test_wc_game_odds_uses_odds_api_above_floor(self):
        d = _router(_states()).route("WORLD_CUP", "game_odds")
        self.assertEqual(d.selected, "odds_api")

    def test_wc_game_odds_skips_when_all_paid_below_floor(self):
        st = _states(
            odds_api=SourceState("odds_api", key_present=True, quota_remaining=10.0),
            sportsgameodds=SourceState("sportsgameodds", key_present=True, quota_remaining=10.0),
        )
        d = _router(st).route("WORLD_CUP", "game_odds")
        self.assertIsNone(d.selected)
        self.assertEqual(d.reason, SKIP_PAID_ODDS)

    def test_kalshi_separate_prediction_market(self):
        d = _router(_states()).route("WORLD_CUP", "prediction_market_prices")
        self.assertEqual(d.selected, "kalshi")  # public, no key required

    def test_nba_player_props_prefers_sgo(self):
        d = _router(_states()).route("NBA", "player_props")
        self.assertEqual(d.selected, "sportsgameodds")

    def test_nba_player_props_falls_back_to_odds_api_when_sgo_no_key(self):
        st = _states(sportsgameodds=SourceState("sportsgameodds", key_present=False))
        d = _router(st).route("NBA", "player_props")
        self.assertEqual(d.selected, "odds_api")
        self.assertEqual(d.skipped[0]["reason"], "no_key")

    def test_recent_failure_cooldown_skips_source(self):
        st = _states(odds_api=SourceState(
            "odds_api", key_present=True, quota_remaining=500.0,
            last_failure_utc=(NOW - timedelta(minutes=20)).isoformat()))
        d = _router(st).route("NBA", "events_schedule")  # [odds_api, sportsgameodds]
        self.assertEqual(d.selected, "sportsgameodds")
        self.assertIn("recent_failure", d.skipped[0]["reason"])

    def test_unsupported_data_type_for_source_is_skipped(self):
        # player_stats only apisports supports (DEFAULT route).
        d = _router(_states()).route("DEFAULT", "player_stats")
        self.assertEqual(d.selected, "apisports")

    def test_unknown_data_type(self):
        d = _router(_states()).route("NBA", "moon_phase")
        self.assertIsNone(d.selected)
        self.assertIn("unknown_data_type", d.reason)

    def test_no_safe_source_reason(self):
        st = _states(kalshi=SourceState("kalshi", key_present=False, blocked_reason="down"))
        d = _router(st).route("NBA", "prediction_market_prices")
        self.assertIsNone(d.selected)
        self.assertEqual(d.reason, SKIP_NO_SAFE_SOURCE)


class TestRecordAndBest(unittest.TestCase):
    def test_record_fetch_shape(self):
        d = _router(_states()).route("WORLD_CUP", "game_odds")
        entry = _router(_states()).record_fetch(
            d, event_id="e1", market_type="h2h", rows=14,
            quota_before=20.0, quota_after=19.0, success=True)
        for k in ("source_selected", "sources_skipped", "reason", "event_id",
                  "rows", "quota_before", "quota_after", "success", "research_only"):
            self.assertIn(k, entry)
        self.assertTrue(entry["success"])
        self.assertEqual(entry["rows"], 14)

    def test_best_source_by_data_type(self):
        best = best_source_by_data_type(CONFIG, _states(), now=NOW)
        self.assertEqual(best["WORLD_CUP"]["events_schedule"], "apisports")
        self.assertEqual(best["NBA"]["player_props"], "sportsgameodds")


if __name__ == "__main__":
    unittest.main()
