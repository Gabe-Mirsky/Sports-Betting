"""Tests for World Cup CLV measurement (pure; no network)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.world_cup_clv import build_world_cup_clv  # noqa: E402


def _row(t, price, line=None, closing=False, outcome="Argentina", market="match_winner_1x2"):
    return {
        "event_id": "e1", "market_type": market, "bookmaker": "draftkings",
        "outcome_name": outcome, "snapshot_time": t, "price": price, "line": line,
        "is_closing_like": closing, "home_team": "Argentina", "away_team": "Brazil",
    }


class TestWorldCupClv(unittest.TestCase):
    def test_no_clv_with_single_snapshot(self):
        df = pd.DataFrame([_row("2026-06-13T00:00:00Z", 2.0)])
        s = build_world_cup_clv(df)
        self.assertFalse(s["clv_ready"])
        self.assertEqual(s["markets_with_clv"], 0)

    def test_clv_two_snapshots_same_line(self):
        df = pd.DataFrame([
            _row("2026-06-13T00:00:00Z", 2.0),
            _row("2026-06-14T11:30:00Z", 1.8, closing=True),
        ])
        s = build_world_cup_clv(df)
        self.assertTrue(s["clv_ready"])
        self.assertEqual(s["markets_with_clv"], 1)
        self.assertEqual(s["price_clv_comparable"], 1)
        pair = s["pairs"][0]
        self.assertAlmostEqual(pair["price_clv"], -0.2, places=6)
        self.assertAlmostEqual(pair["implied_prob_clv"], 0.5 - (1 / 1.8), places=6)
        self.assertFalse(pair["line_moved"])

    def test_line_change_flagged_not_price_compared(self):
        df = pd.DataFrame([
            _row("2026-06-13T00:00:00Z", 1.9, line=2.5, market="total_goals", outcome="Over"),
            _row("2026-06-14T11:30:00Z", 1.9, line=3.5, market="total_goals", outcome="Over", closing=True),
        ])
        s = build_world_cup_clv(df)
        self.assertEqual(s["markets_with_clv"], 1)
        self.assertEqual(s["line_changed"], 1)
        self.assertEqual(s["price_clv_comparable"], 0)
        self.assertIsNone(s["pairs"][0]["price_clv"])

    def test_isolated_from_nba_gates_flag(self):
        s = build_world_cup_clv(pd.DataFrame([_row("2026-06-13T00:00:00Z", 2.0)]))
        self.assertTrue(s["isolated_from_nba_gates"])


if __name__ == "__main__":
    unittest.main()
