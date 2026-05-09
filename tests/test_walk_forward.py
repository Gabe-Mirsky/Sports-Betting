from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from models.walk_forward import available_walk_forward_seasons, walk_forward_predict  # noqa: E402
from models.train_model import available_feature_columns  # noqa: E402


class TestWalkForward(unittest.TestCase):
    def _fake_modeling_data(self) -> pd.DataFrame:
        rows = []
        game_number = 1
        for season in [2018, 2019, 2020]:
            for i in range(12):
                rows.append(
                    {
                        "game_id": f"g{game_number:03d}",
                        "game_date": pd.Timestamp(season, 1, 1) + pd.Timedelta(days=i),
                        "season": season,
                        "home_team_abbr": "AAA",
                        "away_team_abbr": "BBB",
                        "elo_diff_pre": float(i - 6),
                        "elo_home_win_prob": 0.45 + i / 100,
                        "rest_diff": float(i % 3),
                        "home_is_back_to_back": int(i % 2 == 0),
                        "away_is_back_to_back": int(i % 2 == 1),
                        "last_5_win_pct_diff": float(i) / 10,
                        "last_10_win_pct_diff": float(i) / 12,
                        "last_5_point_diff_diff": float(i - 3),
                        "last_10_point_diff_diff": float(i - 4),
                        "season_win_pct_diff": float(i) / 20,
                        "season_avg_margin_diff": float(i - 5),
                        "target_home_win": int(i >= 6),
                    }
                )
                game_number += 1
        return pd.DataFrame(rows)

    def test_available_seasons_start_after_training_start(self) -> None:
        data = self._fake_modeling_data()
        seasons = available_walk_forward_seasons(data, train_start_season=2018)
        self.assertEqual(seasons, [2019, 2020])

    def test_walk_forward_train_end_is_before_test_season(self) -> None:
        data = self._fake_modeling_data()
        predictions, metrics = walk_forward_predict(
            data,
            train_start_season=2018,
            first_test_season=2019,
            last_test_season=2020,
            model_type="logistic_regression",
        )
        self.assertEqual(sorted(predictions["season"].unique().tolist()), [2019, 2020])
        self.assertTrue((predictions["train_end_season"] < predictions["season"]).all())
        self.assertEqual(metrics["folds"][0]["train_end_season"], 2018)
        self.assertEqual(metrics["folds"][1]["train_end_season"], 2019)

    def test_default_features_include_rich_columns_when_available(self) -> None:
        data = self._fake_modeling_data()
        data["last_5_fg_pct_diff"] = 0.01
        data["last_10_ast_diff"] = 1.5

        features = available_feature_columns(data)

        self.assertIn("last_5_fg_pct_diff", features)
        self.assertIn("last_10_ast_diff", features)


if __name__ == "__main__":
    unittest.main()
