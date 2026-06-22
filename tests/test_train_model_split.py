from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

HAS_MODEL_DEPS = all(
    importlib.util.find_spec(package) is not None
    for package in ["joblib", "sklearn"]
)

if HAS_MODEL_DEPS:
    from models.train_model import BASELINE_FEATURE_COLUMNS, time_based_split, train_models  # noqa: E402
else:
    BASELINE_FEATURE_COLUMNS = []


def _modeling_frame() -> pd.DataFrame:
    rows = []
    for season in range(2018, 2026):
        for index in range(10):
            value = float(index - 5)
            row = {
                "game_id": f"{season}-{index}",
                "game_date": pd.Timestamp(season, 10, 1) + pd.Timedelta(days=index),
                "season": season,
                "season_type": "Regular Season",
                "home_team_abbr": "AAA",
                "away_team_abbr": "BBB",
                "target_home_win": int(index % 2 == 0),
            }
            for column in BASELINE_FEATURE_COLUMNS:
                row[column] = 0.5 if column == "elo_home_win_prob" else value
            rows.append(row)
    return pd.DataFrame(rows)


@unittest.skipUnless(HAS_MODEL_DEPS, "model training dependencies are not installed")
class TestTrainModelSplit(unittest.TestCase):
    def test_time_based_split_assigns_exact_holdout_groups(self) -> None:
        train, validation, test = time_based_split(_modeling_frame())

        self.assertEqual(set(train["dataset_split"]), {"train"})
        self.assertEqual(set(validation["dataset_split"]), {"validation"})
        self.assertEqual(set(test["dataset_split"]), {"test"})
        self.assertFalse(train["season"].isin([2024, 2025]).any())
        self.assertEqual(set(validation["season"]), {2024})
        self.assertEqual(set(test["season"]), {2025})

    def test_train_models_selects_on_validation_and_evaluates_test(self) -> None:
        _, metrics, predictions = train_models(_modeling_frame())

        self.assertEqual(metrics["split"]["train_start_season"], 2018)
        self.assertEqual(metrics["split"]["validation_season"], 2024)
        self.assertEqual(metrics["split"]["test_season"], 2025)
        self.assertIn(metrics["best_model"], metrics["models"])
        self.assertIn(metrics["best_model"], metrics["final_test"])
        self.assertEqual(set(predictions["dataset_split"]), {"test"})
        self.assertEqual(set(predictions["season"]), {2025})


if __name__ == "__main__":
    unittest.main()
