"""Tests for the next-action decision logic (synthetic state fixtures only)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_next_action_report import evaluate_next_actions  # noqa: E402


NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


def _state(**overrides) -> dict:
    state = {
        "now_utc": NOW,
        "api_key_present": True,
        "pending_props": 0,
        "settled_props": 0,
        "outcomes_settled_props": 0,
        "unsettled_games": [],
        "upcoming_nba_games": [],
        "actuals_max_date": "2026-06-10",
        "nba_snapshots": 1000,
        "nba_closing_snapshots": 0,
        "nba_clv_markets": 0,
        "total_clv_markets": 100,
        "gate_status": "collection_ready",
        "gate_blockers": ["[settlement] settled_props: value 0 vs threshold 1"],
        "missing_tasks": [],
        "soccer_inactive": True,
        "last_run_id": "r1",
        "last_run_outcome": "success",
        "quota_remaining": 300.0,
    }
    state.update(overrides)
    return state


class NextActionTests(unittest.TestCase):
    def test_missing_api_key_is_top_priority(self) -> None:
        actions = evaluate_next_actions(_state(api_key_present=False))
        self.assertEqual(actions[0]["priority"], 1)
        self.assertIn("ODDS_API_KEY", actions[0]["action"])

    def test_finished_game_with_stale_actuals_wants_download_refresh(self) -> None:
        state = _state(
            pending_props=500,
            unsettled_games=[{
                "canonical_game_key": "k", "label": "SAS @ NYK (2026-06-12)",
                "game_date": "2026-06-12",
                "game_start_utc": NOW - timedelta(hours=6),
            }],
            actuals_max_date="2026-06-08",
        )
        actions = evaluate_next_actions(state)
        top = actions[0]
        self.assertIn("--download", top["command"] or "")
        self.assertEqual(top["priority"], 1)

    def test_in_progress_game_does_not_trigger_refresh(self) -> None:
        state = _state(
            pending_props=500,
            unsettled_games=[{
                "canonical_game_key": "k", "label": "SAS @ NYK", "game_date": "2026-06-11",
                "game_start_utc": NOW - timedelta(hours=1),
            }],
        )
        actions = evaluate_next_actions(state)
        self.assertFalse(any("settlement refresh" in a["action"].lower() for a in actions))

    def test_upcoming_game_with_no_closing_snapshots_suggests_pregame_run(self) -> None:
        state = _state(
            upcoming_nba_games=[{
                "label": "BOS @ LAL (2026-06-11)",
                "game_start_utc": NOW + timedelta(hours=8),
            }],
        )
        actions = evaluate_next_actions(state)
        self.assertTrue(any("pregame collection" in a["action"].lower() for a in actions))

    def test_no_upcoming_game_says_wait(self) -> None:
        actions = evaluate_next_actions(_state())
        self.assertTrue(any("wait for the next nba game" in a["action"].lower() for a in actions))

    def test_missing_tasks_flagged(self) -> None:
        actions = evaluate_next_actions(_state(missing_tasks=["NBA Pregame Prop Collection 1800ET"]))
        recreate = [a for a in actions if "scheduled tasks" in a["action"].lower()]
        self.assertEqual(len(recreate), 1)
        self.assertIn("1800ET", recreate[0]["reason"])

    def test_clv_pair_fixture_suppresses_clv_blocker(self) -> None:
        # Synthetic CLV pair: once NBA CLV markets exist, the CLV explanation
        # action disappears.
        with_clv = evaluate_next_actions(_state(nba_clv_markets=40, nba_closing_snapshots=120))
        self.assertFalse(any("clv is not computable" in a["action"].lower() for a in with_clv))
        without_clv = evaluate_next_actions(_state())
        self.assertTrue(any("clv is not computable" in a["action"].lower() for a in without_clv))

    def test_modeling_ready_announced_without_approving_bets(self) -> None:
        actions = evaluate_next_actions(_state(gate_status="modeling_experiment_ready", gate_blockers=[]))
        modeling = [a for a in actions if "baseline modeling" in a["action"].lower()]
        self.assertEqual(len(modeling), 1)
        self.assertIn("does NOT approve betting", modeling[0]["reason"])

    def test_soccer_inactive_is_lowest_priority(self) -> None:
        actions = evaluate_next_actions(_state())
        soccer = [a for a in actions if "soccer" in a["action"].lower()]
        self.assertEqual(len(soccer), 1)
        self.assertEqual(soccer[0]["priority"], max(a["priority"] for a in actions))


if __name__ == "__main__":
    unittest.main()
