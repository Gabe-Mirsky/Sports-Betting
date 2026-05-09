from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.ensemble import (  # noqa: E402
    build_home_win_ensemble,
    choose_best_weights,
    prepare_home_win_ensemble_frame,
    simplex_weight_grid,
)


class TestHomeWinEnsemble(unittest.TestCase):
    def test_simplex_weight_grid_sums_to_one(self) -> None:
        weights = simplex_weight_grid(3, step=0.5)
        self.assertIn((0.0, 0.5, 0.5), weights)
        self.assertTrue(all(abs(sum(item) - 1.0) < 1e-9 for item in weights))

    def test_prepare_home_win_ensemble_frame_merges_sources(self) -> None:
        base = pd.DataFrame(
            {
                "game_id": ["1"],
                "game_date": ["2024-01-01"],
                "season": [2024],
                "season_type": ["Regular Season"],
                "home_team_abbr": ["BOS"],
                "away_team_abbr": ["NYK"],
                "model_home_win_prob": [0.6],
                "actual_home_win": [1],
            }
        )
        tuned = pd.DataFrame({"game_id": ["1"], "model_home_win_prob": [0.65]})
        margin = pd.DataFrame({"game_id": ["1"], "prob_home_win_from_margin": [0.62]})

        frame = prepare_home_win_ensemble_frame(base, tuned, margin)

        self.assertEqual(len(frame), 1)
        self.assertIn("base_home_win_prob", frame.columns)
        self.assertIn("tuned_home_win_prob", frame.columns)
        self.assertIn("margin_home_win_prob", frame.columns)

    def test_choose_best_weights_prefers_better_column(self) -> None:
        train = pd.DataFrame(
            {
                "actual_home_win": [1, 1, 0, 0],
                "base_home_win_prob": [0.51, 0.51, 0.49, 0.49],
                "tuned_home_win_prob": [0.90, 0.85, 0.10, 0.15],
                "margin_home_win_prob": [0.50, 0.50, 0.50, 0.50],
            }
        )

        weights, _ = choose_best_weights(train, step=0.5)

        self.assertGreaterEqual(weights[1], 0.5)

    def test_build_home_win_ensemble_outputs_model_prob_column(self) -> None:
        rows = []
        for season in [2022, 2023]:
            for index in range(6):
                actual = int(index % 2 == 0)
                rows.append(
                    {
                        "game_id": f"{season}-{index}",
                        "game_date": f"{season}-01-{index + 1:02d}",
                        "season": season,
                        "season_type": "Regular Season",
                        "home_team_abbr": "BOS",
                        "away_team_abbr": "NYK",
                        "actual_home_win": actual,
                        "base_home_win_prob": 0.55 if actual else 0.45,
                        "tuned_home_win_prob": 0.70 if actual else 0.30,
                        "margin_home_win_prob": 0.60 if actual else 0.40,
                    }
                )
        frame = pd.DataFrame(rows)

        predictions, weights, summary = build_home_win_ensemble(frame, min_train_rows=1, weight_step=0.5)

        self.assertIn("model_home_win_prob", predictions.columns)
        self.assertFalse(weights.empty)
        self.assertEqual(summary["rows"], len(frame))


if __name__ == "__main__":
    unittest.main()
