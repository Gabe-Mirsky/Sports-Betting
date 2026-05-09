from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.market_line_audit import build_market_line_coverage  # noqa: E402


class TestMarketLineAudit(unittest.TestCase):
    def test_line_coverage_blocks_sparse_spread_and_total_markets(self) -> None:
        taxonomy = pd.DataFrame(
            [
                {
                    "market_category": "spread_handicap",
                    "line_value": 4.5,
                    "direction": "",
                    "taxonomy_confidence": 0.80,
                },
                {
                    "market_category": "total_points_over_under",
                    "line_value": None,
                    "direction": "over",
                    "taxonomy_confidence": 0.80,
                },
            ]
        )

        coverage, summary = build_market_line_coverage(taxonomy)

        self.assertEqual(int(coverage["rows"].sum()), 2)
        self.assertFalse(summary["spread_ready"])
        self.assertFalse(summary["total_ready"])
        self.assertIn("spread_handicap", summary["blocked_market_types"])

    def test_player_props_are_deferred_even_when_lines_parse(self) -> None:
        taxonomy = pd.DataFrame(
            [
                {
                    "market_category": "player_points_rebounds_assists",
                    "line_value": 1.0,
                    "direction": "over",
                    "taxonomy_confidence": 0.85,
                }
                for _ in range(60)
            ]
        )

        coverage, summary = build_market_line_coverage(taxonomy)
        player_row = coverage.set_index("market_type").loc["player_points_rebounds_assists"]

        self.assertTrue(player_row["line_extraction_ready"])
        self.assertFalse(player_row["ready_for_market_specific_backtest"])
        self.assertIn("player_points_rebounds_assists", summary["blocked_market_types"])
        self.assertEqual(
            player_row["blocked_reason"],
            "player_props_deferred_until_spread_total_models_are_ready",
        )


if __name__ == "__main__":
    unittest.main()
