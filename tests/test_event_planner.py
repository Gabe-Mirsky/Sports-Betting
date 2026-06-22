"""Time-window logic tests for the generic event planner (sport-agnostic)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.event_planner import (  # noqa: E402
    ACTION_CLOSING, ACTION_EARLY, ACTION_ERROR, ACTION_POSTGAME, ACTION_SKIP,
    STATUS_ENDED, STATUS_IN_PROGRESS, STATUS_STARTING_SOON, STATUS_UNKNOWN,
    EventWindows, actions_due, plan_event, plan_events,
)

NOW = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
W = EventWindows()  # defaults: closing 60/grace 10, early 24-48h, post 150-420


def _evt(minutes_from_now, **kw):
    start = NOW + timedelta(minutes=minutes_from_now)
    base = {
        "event_id": kw.pop("event_id", "e1"), "league": "WORLD_CUP",
        "sport_key": "soccer_fifa_world_cup", "home_team": "ARG", "away_team": "BRA",
        "event_start_time": start.isoformat(),
    }
    base.update(kw)
    return base


class TestPlanEvent(unittest.TestCase):
    def _action(self, minutes, **kw):
        return plan_event(_evt(minutes, **kw), NOW, W)

    def test_closing_within_60_min(self):
        p = self._action(30)
        self.assertEqual(p["recommended_action"], ACTION_CLOSING)
        self.assertEqual(p["event_status"], STATUS_STARTING_SOON)

    def test_closing_boundary_at_60(self):
        self.assertEqual(self._action(60)["recommended_action"], ACTION_CLOSING)

    def test_closing_grace_just_after_kickoff(self):
        # 5 minutes after kickoff is still inside the 10-min grace.
        self.assertEqual(self._action(-5)["recommended_action"], ACTION_CLOSING)

    def test_between_closing_and_early_is_skip(self):
        p = self._action(90)
        self.assertEqual(p["recommended_action"], ACTION_SKIP)

    def test_early_window_24_to_48h(self):
        p = self._action(36 * 60)  # 36h out
        self.assertEqual(p["recommended_action"], ACTION_EARLY)

    def test_early_window_respects_allow_early_false(self):
        p = plan_event(_evt(36 * 60), NOW, W, allow_early=False)
        self.assertEqual(p["recommended_action"], ACTION_SKIP)

    def test_too_far_future_is_skip(self):
        self.assertEqual(self._action(72 * 60)["recommended_action"], ACTION_SKIP)

    def test_postgame_results_after_match(self):
        p = self._action(-200)  # 200 min after kickoff
        self.assertEqual(p["recommended_action"], ACTION_POSTGAME)
        self.assertEqual(p["event_status"], STATUS_ENDED)

    def test_in_progress_is_skip(self):
        p = self._action(-60)  # 60 min after kickoff, still playing
        self.assertEqual(p["recommended_action"], ACTION_SKIP)
        self.assertEqual(p["event_status"], STATUS_IN_PROGRESS)

    def test_ended_long_ago_is_skip(self):
        self.assertEqual(self._action(-500)["recommended_action"], ACTION_SKIP)

    def test_missing_start_is_error(self):
        p = plan_event({"event_id": "x", "event_start_time": None}, NOW, W)
        self.assertEqual(p["recommended_action"], ACTION_ERROR)
        self.assertEqual(p["event_status"], STATUS_UNKNOWN)

    def test_completed_flag_forces_postgame(self):
        p = plan_event(_evt(-30, completed=True), NOW, W)  # would be in_progress otherwise
        self.assertEqual(p["recommended_action"], ACTION_POSTGAME)

    def test_minutes_until_event_sign(self):
        self.assertEqual(self._action(45)["minutes_until_event"], 45.0)
        self.assertEqual(self._action(-45)["minutes_until_event"], -45.0)


class TestActionsDue(unittest.TestCase):
    def test_grouping_excludes_skip(self):
        events = [_evt(30, event_id="a"), _evt(90, event_id="b"), _evt(-200, event_id="c")]
        planned = plan_events(events, NOW, W)
        due = actions_due(planned)
        self.assertEqual([p["event_id"] for p in due[ACTION_CLOSING]], ["a"])
        self.assertEqual([p["event_id"] for p in due[ACTION_POSTGAME]], ["c"])
        self.assertNotIn("b", [p["event_id"] for grp in due.values() for p in grp])


if __name__ == "__main__":
    unittest.main()
