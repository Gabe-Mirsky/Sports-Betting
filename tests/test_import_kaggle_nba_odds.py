from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from import_kaggle_nba_odds import import_file  # noqa: E402


class TestImportKaggleNbaOdds(unittest.TestCase):
    def test_imports_kaggle_team_row_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nba odds 2022-23.csv"
            pd.DataFrame(
                [
                    {
                        "Date": 1025,
                        "Rot": 501,
                        "VH": "V",
                        "Team": "GoldenState",
                        "1st": 25,
                        "2nd": 25,
                        "3rd": 25,
                        "4th": 25,
                        "Final": 100,
                        "Open": "",
                        "Close": 221.5,
                        "ML": 120,
                        "2H": "",
                    },
                    {
                        "Date": 1025,
                        "Rot": 502,
                        "VH": "H",
                        "Team": "LALakers",
                        "1st": 30,
                        "2nd": 30,
                        "3rd": 30,
                        "4th": 30,
                        "Final": 120,
                        "Open": "",
                        "Close": -2.5,
                        "ML": -140,
                        "2H": "",
                    },
                ]
            ).to_csv(path, index=False)

            rows, warnings = import_file(path)

        self.assertEqual(warnings, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["game_date"], "2022-10-25")
        self.assertEqual(rows[0]["home_team"], "Los Angeles Lakers")
        self.assertEqual(rows[0]["away_team"], "Golden State Warriors")
        self.assertEqual(rows[0]["home_moneyline"], -140)
        self.assertEqual(rows[0]["away_moneyline"], 120)
        self.assertEqual(rows[0]["home_score"], 120)
        self.assertEqual(rows[0]["away_score"], 100)
        self.assertEqual(rows[0]["spread"], -2.5)
        self.assertEqual(rows[0]["total"], 221.5)


if __name__ == "__main__":
    unittest.main()
