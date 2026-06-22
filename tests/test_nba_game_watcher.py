"""Reliability tests for the NBA game watcher's duplicate-prevention logic.

These cover the run-log dedup rules that keep the every-10-minute watcher from
firing the same action twice while still retrying transient failures. No network,
no subprocesses: the pure decision helpers only.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import nba_game_watcher as watcher  # noqa: E402


class TestWatcherDedup(unittest.TestCase):
    def test_action_state_counts_success_and_attempts(self):
        history = [
            {"game_id": "g1", "action": "pregame", "status": "failed"},
            {"game_id": "g1", "action": "pregame", "status": "success"},
            {"game_id": "g1", "action": "settle", "status": "failed"},
            {"game_id": "g2", "action": "pregame", "status": "dry_run"},  # ignored
        ]
        state = watcher.action_state(history)
        self.assertEqual(state[("g1", "pregame")], {"success": 1, "attempts": 2})
        self.assertEqual(state[("g1", "settle")], {"success": 0, "attempts": 1})
        # dry_run rows do not count as attempts or successes.
        self.assertEqual(state[("g2", "pregame")], {"success": 0, "attempts": 0})

    def test_first_time_action_is_needed(self):
        self.assertTrue(watcher.needs_action({}, "g1", "pregame", max_attempts=3))

    def test_no_repeat_after_success(self):
        state = watcher.action_state(
            [{"game_id": "g1", "action": "pregame", "status": "success"}]
        )
        self.assertFalse(watcher.needs_action(state, "g1", "pregame", max_attempts=3))

    def test_failures_retry_until_cap(self):
        # One failure so far: still retryable under a cap of 3.
        state = watcher.action_state(
            [{"game_id": "g1", "action": "settle", "status": "failed"}]
        )
        self.assertTrue(watcher.needs_action(state, "g1", "settle", max_attempts=3))

    def test_stops_after_max_attempts(self):
        state = watcher.action_state(
            [{"game_id": "g1", "action": "settle", "status": "failed"}] * 3
        )
        self.assertFalse(watcher.needs_action(state, "g1", "settle", max_attempts=3))

    def test_success_overrides_remaining_attempts(self):
        # A success anywhere in history means done, even with prior failures.
        state = watcher.action_state([
            {"game_id": "g1", "action": "pregame", "status": "failed"},
            {"game_id": "g1", "action": "pregame", "status": "success"},
        ])
        self.assertFalse(watcher.needs_action(state, "g1", "pregame", max_attempts=5))


class TestWatcherWiring(unittest.TestCase):
    def test_action_constants(self):
        self.assertEqual(watcher.ACTION_PREGAME, "pregame")
        self.assertEqual(watcher.ACTION_SETTLE, "settle")

    def test_launcher_bat_exists(self):
        self.assertTrue((PROJECT_ROOT / "run_nba_game_watcher.bat").exists())


if __name__ == "__main__":
    unittest.main()
