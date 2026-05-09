from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from models.market_type_models import (  # noqa: E402
    build_market_type_probability_grid,
    prepare_margin_total_dataset,
    probability_home_spread_covers,
    probability_margin_exceeds,
    probability_total_over,
    summarize_market_type_calibration,
)


class TestMarketTypeModels(unittest.TestCase):
    def test_probability_engines_move_in_expected_direction(self) -> None:
        self.assertGreater(probability_margin_exceeds(5.0, 12.0, 0.0), 0.5)
        self.assertLess(probability_margin_exceeds(-5.0, 12.0, 0.0), 0.5)
        self.assertGreater(probability_home_spread_covers(8.0, 12.0, -4.5), 0.5)
        self.assertLess(probability_home_spread_covers(1.0, 12.0, -4.5), 0.5)
        self.assertGreater(probability_total_over(228.0, 18.0, 220.5), 0.5)
        self.assertLess(probability_total_over(212.0, 18.0, 220.5), 0.5)

    def test_prepare_margin_total_dataset_joins_scores(self) -> None:
        modeling = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2025-01-01",
                    "season": 2024,
                    "elo_diff_pre": 10.0,
                    "elo_home_win_prob": 0.6,
                }
            ]
        )
        games = pd.DataFrame([{"game_id": "1", "home_points": 110, "away_points": 100}])

        output = prepare_margin_total_dataset(modeling, games_df=games)

        self.assertEqual(output.loc[0, "target_home_margin"], 10)
        self.assertEqual(output.loc[0, "target_total_points"], 210)

    def test_market_type_calibration_grid_contains_spread_and_total_rows(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2025-01-01",
                    "season": 2024,
                    "target_home_margin": 6.0,
                    "target_total_points": 221.0,
                    "pred_home_margin": 4.0,
                    "pred_total_points": 219.0,
                    "margin_residual_std_train": 12.0,
                    "total_residual_std_train": 18.0,
                }
            ]
        )

        grid = build_market_type_probability_grid(predictions, margin_thresholds=[3.5], total_lines=[220.5])
        calibration, summary = summarize_market_type_calibration(grid)

        self.assertEqual(set(grid["market_type"]), {"spread_handicap", "total_points_over_under"})
        self.assertFalse(calibration.empty)
        self.assertEqual(summary["rows"], 2)


if __name__ == "__main__":
    unittest.main()
