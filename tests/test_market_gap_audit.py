from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.market_gap_audit import build_market_gap_detail, build_market_gap_segment_summary  # noqa: E402


class TestMarketGapAudit(unittest.TestCase):
    def test_gap_detail_marks_when_kalshi_is_closer_than_model(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2026-01-01",
                    "market_ticker": "MKT-1",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.55,
                    "market_yes_prob": 0.80,
                    "actual_yes_win": True,
                    "is_playoffs": False,
                    "volume": 100,
                }
            ]
        )
        modeling = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "rest_diff": 2,
                    "player_top8_minutes_gap_last_game_diff": 25,
                }
            ]
        )

        detail = build_market_gap_detail(predictions, modeling)
        row = detail.iloc[0]

        self.assertGreater(row["kalshi_edge_over_model"], 0)
        self.assertEqual(row["yes_rest_context"], "yes_2plus_days_more_rest")
        self.assertEqual(row["yes_player_availability_proxy"], "yes_more_rotation_gap")

    def test_segment_summary_reports_accuracy_by_segment(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2026-01-01",
                    "market_ticker": "MKT-1",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.45,
                    "market_yes_prob": 0.70,
                    "actual_yes_win": True,
                    "is_playoffs": False,
                    "volume": 100,
                },
                {
                    "game_id": "2",
                    "game_date": "2026-01-02",
                    "market_ticker": "MKT-2",
                    "home_team_abbr": "NYK",
                    "away_team_abbr": "BOS",
                    "yes_team_abbr": "BOS",
                    "model_yes_prob": 0.70,
                    "market_yes_prob": 0.45,
                    "actual_yes_win": True,
                    "is_playoffs": False,
                    "volume": 100,
                },
            ]
        )

        detail = build_market_gap_detail(predictions)
        segments = build_market_gap_segment_summary(detail)
        agreement = segments[segments["segment"].eq("model_market_pick_agreement")]

        self.assertIn("opposite_pick", agreement["value"].tolist())
        self.assertTrue((agreement["rows"] > 0).all())


if __name__ == "__main__":
    unittest.main()
