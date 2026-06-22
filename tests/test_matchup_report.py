from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from reports.matchup_prediction_report import (  # noqa: E402
    build_backtest_report,
    build_today_predictions_report,
)

_REQUIRED_CSV_COLUMNS = [
    "id",
    "sport",
    "league",
    "game_date",
    "team_a",
    "team_b",
    "prob_team_a_win",
    "prob_draw",
    "prob_team_b_win",
    "predicted_outcome",
    "confidence_level",
    "data_quality",
    "key_reasons",
    "main_risks",
    "data_quality_warnings",
    "model_version",
]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixture_id": "fx1",
                "sport": "soccer",
                "league": "international_friendly",
                "game_date": "2026-06-20T19:00:00",
                "team_a": "Japan",
                "team_b": "Tunisia",
                "prob_team_a_win": 0.46,
                "prob_draw": 0.27,
                "prob_team_b_win": 0.27,
                "predicted_outcome": "Japan win",
                "predicted_side": "team_a",
                "confidence_score": 0.19,
                "competition_type": "friendly",
                "neutral_site": 1,
                "elo_diff": 40.0,
                "recent_win_rate_diff_5": 0.2,
                "min_recent_games": 12,
                "team_a_recent_games": 12,
                "team_b_recent_games": 12,
                "injury_data_present": 0,
                "team_a_injury_stale": False,
                "team_b_injury_stale": False,
                "team_a_key_players_out": 0,
                "team_b_key_players_out": 2,
                "model_version": "matchup_baseline_v1",
            }
        ]
    )


class TestPredictionsReport(unittest.TestCase):
    def test_csv_and_json_created_with_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            build_today_predictions_report(_predictions(), out_dir)

            csv_path = out_dir / "matchup_predictions_today.csv"
            json_path = out_dir / "matchup_predictions_today.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            df = pd.read_csv(csv_path)
            for column in _REQUIRED_CSV_COLUMNS:
                self.assertIn(column, df.columns)

            records = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIsInstance(records, list)
            record = records[0]
            for key in ("fixture_id", "prob_team_a_win", "prob_draw", "prob_team_b_win",
                        "predicted_outcome", "confidence_level", "data_quality",
                        "key_reasons", "main_risks", "data_quality_warnings", "model_version"):
                self.assertIn(key, record)
            self.assertIsInstance(record["key_reasons"], list)
            # Missing injury data should be surfaced as a warning.
            self.assertTrue(any("injury" in w.lower() for w in record["data_quality_warnings"]))

    def test_backtest_report_files_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            metrics = {"n_games": 10, "accuracy": 0.5, "log_loss": 1.0, "brier_score": 0.6}
            buckets = pd.DataFrame(
                [{"prob_bucket": "0.5-0.6", "n_games": 10, "mean_predicted_prob": 0.55,
                  "actual_win_rate": 0.5, "calibration_gap": 0.05}]
            )
            build_backtest_report(metrics, buckets, out_dir)
            self.assertTrue((out_dir / "matchup_model_backtest.json").exists())
            self.assertTrue((out_dir / "matchup_model_backtest_by_bucket.csv").exists())
            loaded = json.loads((out_dir / "matchup_model_backtest.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["n_games"], 10)


if __name__ == "__main__":
    unittest.main()
