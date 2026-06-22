from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from reports.availability_gaps import build_availability_gap_report  # noqa: E402


class TestAvailabilityGaps(unittest.TestCase):
    def test_gap_report_excludes_completed_statuses(self) -> None:
        template = pd.DataFrame(
            [
                {
                    "report_date": "2026-01-01",
                    "game_date": "2026-01-01",
                    "game_id": "g1",
                    "team_abbr": "NYK",
                    "opponent_abbr": "BOS",
                    "home_away": "home",
                    "player_id": "1",
                    "player_name": "Starter",
                    "status": "",
                    "impact_weight": 30,
                },
                {
                    "report_date": "2026-01-01",
                    "game_date": "2026-01-01",
                    "game_id": "g1",
                    "team_abbr": "NYK",
                    "opponent_abbr": "BOS",
                    "home_away": "home",
                    "player_id": "2",
                    "player_name": "Bench",
                    "status": "",
                    "impact_weight": 10,
                },
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "report_date": "2026-01-01",
                    "game_date": "2026-01-01",
                    "team_abbr": "NYK",
                    "player_name": "Starter",
                    "status": "available",
                    "impact_weight": 30,
                }
            ]
        )

        gaps, summary = build_availability_gap_report(template, availability, high_impact_minutes=20)

        self.assertEqual(gaps["player_name"].tolist(), ["Bench"])
        self.assertEqual(summary["missing_rows"], 1)
        self.assertEqual(summary["high_impact_missing_rows"], 0)


if __name__ == "__main__":
    unittest.main()
