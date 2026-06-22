from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quality.team_availability_validation import (  # noqa: E402
    STATUS_WARNING,
    build_team_availability_validation_report,
    save_team_availability_validation_report,
)


class TestTeamAvailabilityValidation(unittest.TestCase):
    def _fixtures(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fixture_id": "fx1",
                    "game_date": "2026-06-20",
                    "team_a": "Japan",
                    "team_b": "Tunisia",
                    "sport": "soccer",
                    "league": "international",
                    "competition_type": "FIFA World Cup",
                }
            ]
        )

    def test_computes_fixture_team_coverage_and_missing_warning(self) -> None:
        injuries = pd.DataFrame(
            [
                {
                    "team": "Japan",
                    "player_name": "Winger",
                    "status": "questionable",
                    "importance_score": 0.8,
                    "last_updated": "2026-06-20",
                    "source": "manual",
                }
            ]
        )

        report = build_team_availability_validation_report(
            self._fixtures(),
            injuries,
            raw_injuries=injuries,
            as_of_date="2026-06-20",
        )

        self.assertEqual(report["overall_status"], STATUS_WARNING)
        self.assertEqual(report["coverage"]["total_fixture_teams"], 2)
        self.assertEqual(report["coverage"]["fixture_teams_with_availability"], 1)
        self.assertEqual(report["coverage"]["coverage_percentage"], 50.0)
        self.assertEqual(report["coverage"]["missing_teams"], ["Tunisia"])
        self.assertTrue(any("No availability rows" in w for w in report["warnings"]))

    def test_warns_for_stale_invalid_status_and_unknown_player_rows(self) -> None:
        injuries = pd.DataFrame(
            [
                {
                    "team": "Japan",
                    "player_name": "Unknown player 1",
                    "status": "banana",
                    "importance_score": "",
                    "last_updated": "2026-06-17",
                    "source": "manual",
                },
                {
                    "team": "Tunisia",
                    "player_name": "",
                    "status": "unknown",
                    "importance_score": 0.25,
                    "last_updated": "2026-06-20",
                    "source": "manual",
                },
            ]
        )

        report = build_team_availability_validation_report(
            self._fixtures(),
            injuries,
            raw_injuries=injuries,
            as_of_date="2026-06-20",
        )

        injury = report["injury_data"]
        self.assertEqual(injury["invalid_status_rows"], 1)
        self.assertEqual(injury["stale_rows_older_than_48h"], 1)
        self.assertEqual(injury["missing_player_names"], 1)
        self.assertEqual(injury["unknown_player_rows"], 1)
        self.assertEqual(injury["missing_importance_scores"], 1)
        self.assertTrue(any("invalid availability status" in w for w in report["warnings"]))

    def test_writes_json_and_markdown_reports(self) -> None:
        injuries = pd.DataFrame(
            [
                {
                    "team": "Japan",
                    "player_name": "Winger",
                    "status": "available",
                    "importance_score": 0.7,
                    "last_updated": "2026-06-20",
                    "source": "manual",
                },
                {
                    "team": "Tunisia",
                    "player_name": "Forward",
                    "status": "out",
                    "importance_score": 0.9,
                    "last_updated": "2026-06-20",
                    "source": "manual",
                },
            ]
        )
        report = build_team_availability_validation_report(
            self._fixtures(),
            injuries,
            raw_injuries=injuries,
            as_of_date="2026-06-20",
        )

        with tempfile.TemporaryDirectory() as tmp:
            json_path, md_path = save_team_availability_validation_report(report, tmp)

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("Team Availability Validation", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
