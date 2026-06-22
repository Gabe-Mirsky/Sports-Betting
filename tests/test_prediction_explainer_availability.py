from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.prediction_explainer import detect_data_quality_warnings, detect_prediction_risks  # noqa: E402


class TestPredictionExplainerAvailability(unittest.TestCase):
    def test_warnings_name_missing_stale_and_manual_availability(self) -> None:
        row = pd.Series(
            {
                "team_a": "Ecuador",
                "team_b": "Tunisia",
                "team_a_availability_present": 0,
                "team_b_availability_present": 1,
                "team_b_injury_stale": True,
                "team_b_availability_source": "manual",
                "team_b_availability_manual": True,
                "team_b_key_players_out": 1,
                "team_a_recent_games": 10,
                "team_b_recent_games": 10,
                "competition_type": "FIFA World Cup",
                "model_backtested": True,
            }
        )

        warnings = detect_data_quality_warnings(row)
        risks = detect_prediction_risks(row)

        self.assertIn("No availability data for Ecuador.", warnings)
        self.assertIn("Availability data for Tunisia is older than 48 hours.", warnings)
        self.assertIn("Injury impact for Tunisia is based on manual availability data.", warnings)
        self.assertIn("Tunisia has 1 key player(s) listed out.", warnings)
        self.assertIn("No availability data for Ecuador.", risks)


if __name__ == "__main__":
    unittest.main()
