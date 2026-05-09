from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from models.train_model import BASELINE_FEATURE_COLUMNS  # noqa: E402
from models.tuning import available_feature_sets, tune_home_win_model  # noqa: E402


def _tiny_modeling_frame() -> pd.DataFrame:
    rows = []
    for season in [2020, 2021, 2022]:
        for idx in range(8):
            home_strength = (idx % 4) - 1.5
            target = int(home_strength + (season - 2020) * 0.1 > 0)
            row = {
                "game_id": f"{season}-{idx}",
                "game_date": f"{season}-01-{idx + 1:02d}",
                "season": season,
                "season_type": "Regular Season",
                "home_team_abbr": "AAA",
                "away_team_abbr": "BBB",
                "target_home_win": target,
                "home_win": target,
            }
            for column in BASELINE_FEATURE_COLUMNS:
                row[column] = home_strength if column != "elo_home_win_prob" else 0.5 + home_strength * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


class TestModelTuning(unittest.TestCase):
    def test_available_feature_sets_include_baseline(self) -> None:
        feature_sets = available_feature_sets(_tiny_modeling_frame())

        self.assertIn("baseline", feature_sets)
        self.assertEqual(feature_sets["baseline"], BASELINE_FEATURE_COLUMNS)

    def test_tune_home_win_model_returns_ranked_results(self) -> None:
        results, predictions, summary, top = tune_home_win_model(
            _tiny_modeling_frame(),
            train_start_season=2020,
            first_test_season=2021,
            random_seed=42,
        )

        self.assertFalse(results.empty)
        self.assertFalse(predictions.empty)
        self.assertEqual(summary["best_model_name"], results.iloc[0]["model_name"])
        self.assertFalse(top.empty)


if __name__ == "__main__":
    unittest.main()
