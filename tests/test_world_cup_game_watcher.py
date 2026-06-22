"""World Cup watcher: dedup + wiring tests (no network).

Integration (real event listing + decisions) is exercised by the live
`world_cup_game_watcher.py --dry-run` command; here we test the importable
pieces: the shared dedup run-log keyed by event_id, and that the watcher's
launcher/config/setup files exist and the module imports cleanly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.watcher_run_log import action_state, needs_action  # noqa: E402
from data.event_planner import ACTION_CLOSING, ACTION_EARLY, ACTION_POSTGAME  # noqa: E402
import world_cup_game_watcher as wcw  # noqa: E402


class TestWatcherDedupByEvent(unittest.TestCase):
    def test_action_state_keys_on_event_id(self):
        history = [
            {"event_id": "wc1", "action": "closing", "status": "failed"},
            {"event_id": "wc1", "action": "closing", "status": "success"},
            {"event_id": "wc1", "action": "results", "status": "dry_run"},  # ignored
        ]
        state = action_state(history)
        self.assertEqual(state[("wc1", "closing")], {"success": 1, "attempts": 2})
        self.assertEqual(state[("wc1", "results")], {"success": 0, "attempts": 0})

    def test_no_repeat_after_success(self):
        state = action_state([{"event_id": "wc1", "action": "closing", "status": "success"}])
        self.assertFalse(needs_action(state, "wc1", "closing", max_attempts=3))

    def test_retry_until_cap(self):
        state = action_state([{"event_id": "wc1", "action": "results", "status": "failed"}] * 2)
        self.assertTrue(needs_action(state, "wc1", "results", max_attempts=3))
        state3 = action_state([{"event_id": "wc1", "action": "results", "status": "failed"}] * 3)
        self.assertFalse(needs_action(state3, "wc1", "results", max_attempts=3))

    def test_first_time_needs_action(self):
        self.assertTrue(needs_action({}, "wc-new", "closing", max_attempts=3))


class TestWatcherWiring(unittest.TestCase):
    def test_action_label_map(self):
        self.assertEqual(wcw.ACTION_LABEL[ACTION_CLOSING], "closing")
        self.assertEqual(wcw.ACTION_LABEL[ACTION_EARLY], "early")
        self.assertEqual(wcw.ACTION_LABEL[ACTION_POSTGAME], "results")

    def test_launcher_and_config_exist(self):
        self.assertTrue((PROJECT_ROOT / "run_world_cup_game_watcher.bat").exists())
        self.assertTrue((PROJECT_ROOT / "config" / "world_cup_collection.yaml").exists())
        self.assertTrue((PROJECT_ROOT / "scripts" / "setup_world_cup_game_watcher_task.ps1").exists())


if __name__ == "__main__":
    unittest.main()
