from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.player_data_audit import build_player_data_audit  # noqa: E402


class TestPlayerDataAudit(unittest.TestCase):
    def test_audit_reports_player_feature_coverage(self) -> None:
        games = pd.DataFrame(
            [
                {"game_id": "g1", "game_date": "2024-01-01", "season": 2023},
                {"game_id": "g2", "game_date": "2024-01-02", "season": 2023},
            ]
        )
        modeling = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "season": 2023,
                    "player_prior_games_last10_diff": 4.0,
                    "player_top8_points_last10_diff": 10.0,
                },
                {
                    "game_id": "g2",
                    "season": 2023,
                    "player_prior_games_last10_diff": None,
                    "player_top8_points_last10_diff": None,
                },
            ]
        )
        player_logs = pd.DataFrame(
            [
                {
                    "GAME_ID": "g0",
                    "GAME_DATE": "2023-12-30",
                    "TEAM_ID": 1,
                    "PLAYER_ID": "p1",
                    "season_start_year": 2023,
                    "source_file": "player_game_log_2023.parquet",
                }
            ]
        )

        summary, feature_coverage, season_coverage = build_player_data_audit(games, modeling, player_logs)

        self.assertEqual(summary["raw_player_log_rows"], 1)
        self.assertEqual(summary["player_feature_columns_present"], 2)
        self.assertAlmostEqual(summary["player_feature_row_coverage"], 0.5)
        self.assertIn("low_player_feature_coverage", summary["warnings"])
        self.assertIn("player_rotation", set(feature_coverage["feature_group"]))
        self.assertAlmostEqual(season_coverage.loc[0, "player_feature_coverage"], 0.5)

    def test_audit_handles_missing_player_logs(self) -> None:
        summary, feature_coverage, season_coverage = build_player_data_audit(
            games=pd.DataFrame(),
            modeling=pd.DataFrame(),
            player_logs=pd.DataFrame(),
        )

        self.assertEqual(summary["status"], "not_ready")
        self.assertIn("no_raw_player_logs", summary["warnings"])
        self.assertFalse(feature_coverage.empty)
        self.assertTrue(season_coverage.empty)


if __name__ == "__main__":
    unittest.main()
