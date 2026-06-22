"""Tests for World Cup easy-market settlement logic (pure; no network)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.world_cup_results import (  # noqa: E402
    parse_scores_payload, settle_market, settle_snapshots,
)

HOME, AWAY = "Argentina", "Brazil"


class TestSettleMarket(unittest.TestCase):
    def test_1x2_home_win(self):
        self.assertEqual(settle_market("match_winner_1x2", HOME, None, HOME, AWAY, 2, 1), "won")
        self.assertEqual(settle_market("match_winner_1x2", AWAY, None, HOME, AWAY, 2, 1), "lost")
        self.assertEqual(settle_market("match_winner_1x2", "Draw", None, HOME, AWAY, 2, 1), "lost")

    def test_1x2_draw(self):
        self.assertEqual(settle_market("match_winner_1x2", "Draw", None, HOME, AWAY, 1, 1), "won")
        self.assertEqual(settle_market("match_winner_1x2", HOME, None, HOME, AWAY, 1, 1), "lost")

    def test_totals_over_under_push(self):
        self.assertEqual(settle_market("total_goals", "Over", 2.5, HOME, AWAY, 2, 1), "won")   # 3>2.5
        self.assertEqual(settle_market("total_goals", "Under", 2.5, HOME, AWAY, 2, 1), "lost")
        self.assertEqual(settle_market("total_goals", "Over", 3.0, HOME, AWAY, 2, 1), "push")  # 3==3
        self.assertEqual(settle_market("total_goals", "Under", 3.0, HOME, AWAY, 2, 1), "push")
        self.assertEqual(settle_market("total_goals", "Under", 4.5, HOME, AWAY, 2, 1), "won")

    def test_btts(self):
        self.assertEqual(settle_market("both_teams_to_score", "Yes", None, HOME, AWAY, 2, 1), "won")
        self.assertEqual(settle_market("both_teams_to_score", "No", None, HOME, AWAY, 2, 1), "lost")
        self.assertEqual(settle_market("both_teams_to_score", "Yes", None, HOME, AWAY, 2, 0), "lost")
        self.assertEqual(settle_market("both_teams_to_score", "No", None, HOME, AWAY, 2, 0), "won")

    def test_unsupported_market(self):
        self.assertEqual(settle_market("player_goals", "Messi Over", 0.5, HOME, AWAY, 2, 1), "unsupported")

    def test_missing_score_is_unsupported(self):
        self.assertEqual(settle_market("match_winner_1x2", HOME, None, HOME, AWAY, None, None), "unsupported")


class TestParseAndSettleFrame(unittest.TestCase):
    def test_parse_scores_payload(self):
        payload = [{"id": "e1", "completed": True, "home_team": HOME, "away_team": AWAY,
                    "scores": [{"name": HOME, "score": "2"}, {"name": AWAY, "score": "1"}]}]
        parsed = parse_scores_payload(payload)
        self.assertEqual(parsed["e1"]["home_score"], 2)
        self.assertTrue(parsed["e1"]["completed"])

    def test_settle_snapshots_marks_pending_for_unknown_event(self):
        snaps = pd.DataFrame([
            {"event_id": "e1", "market_type": "match_winner_1x2", "outcome_name": HOME,
             "line": None, "home_team": HOME, "away_team": AWAY},
            {"event_id": "e2", "market_type": "match_winner_1x2", "outcome_name": HOME,
             "line": None, "home_team": HOME, "away_team": AWAY},
        ])
        scores = {"e1": {"home_team": HOME, "away_team": AWAY, "home_score": 2,
                         "away_score": 1, "completed": True}}
        out = settle_snapshots(snaps, scores)
        by_event = dict(zip(out["event_id"], out["result"]))
        self.assertEqual(by_event["e1"], "won")
        self.assertEqual(by_event["e2"], "pending")  # no score for e2


if __name__ == "__main__":
    unittest.main()
