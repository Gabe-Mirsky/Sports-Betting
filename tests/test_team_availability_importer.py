from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.team_availability_importer import import_team_availability  # noqa: E402


class TestTeamAvailabilityImporter(unittest.TestCase):
    def test_normalizes_status_importance_aliases_and_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "availability.csv"
            aliases = root / "aliases.csv"
            output = root / "injuries.csv"

            pd.DataFrame(
                [
                    {
                        "team": "Deutschland",
                        "player_name": "Forward",
                        "status": "GTD",
                        "expected_minutes_or_role": "star",
                        "last_updated": "",
                        "source": "manual",
                    },
                    {
                        "team": "Japan",
                        "player_name": "",
                        "status": "active",
                        "expected_minutes_or_role": "bench",
                        "last_updated": "2026-06-19",
                    },
                    {
                        "team": "",
                        "player_name": "Dropped",
                        "status": "out",
                        "expected_minutes_or_role": "starter",
                    },
                ]
            ).to_csv(source, index=False)
            pd.DataFrame(
                [{"canonical_team": "Germany", "alias": "Deutschland"}]
            ).to_csv(aliases, index=False)

            imported, summary = import_team_availability(
                source,
                output,
                aliases_path=aliases,
                as_of_date="2026-06-20",
            )

            self.assertTrue(output.exists())
            self.assertEqual(summary.rows_read, 3)
            self.assertEqual(summary.rows_dropped_blank_team, 1)
            self.assertEqual(summary.rows_written, 2)
            self.assertEqual(set(imported["team"]), {"Germany", "Japan"})
            germany = imported[imported["team"] == "Germany"].iloc[0]
            self.assertEqual(germany["status"], "questionable")
            self.assertAlmostEqual(float(germany["importance_score"]), 1.0)
            self.assertEqual(pd.to_datetime(germany["last_updated"]).date().isoformat(), "2026-06-20")
            japan = imported[imported["team"] == "Japan"].iloc[0]
            self.assertEqual(japan["status"], "available")
            self.assertAlmostEqual(float(japan["importance_score"]), 0.15)
            self.assertFalse({"odds", "price", "clv"} & set(c.lower() for c in imported.columns))

    def test_drop_unknown_player_rows_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "availability.csv"
            output = root / "injuries.csv"
            pd.DataFrame(
                [
                    {"team": "Japan", "player_name": "", "status": "unknown"},
                    {"team": "Japan", "player_name": "Known", "status": "out"},
                ]
            ).to_csv(source, index=False)

            imported, summary = import_team_availability(
                source,
                output,
                as_of_date="2026-06-20",
                drop_unknown_player_rows=True,
            )

        self.assertEqual(summary.rows_dropped_unknown_player, 1)
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported.iloc[0]["player_name"], "Known")


if __name__ == "__main__":
    unittest.main()
