from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.seasons import assign_dataset_split, current_nba_season_start_year, nba_season_display_label  # noqa: E402


class TestSeasons(unittest.TestCase):
    def test_may_2026_is_2025_26_nba_season(self) -> None:
        self.assertEqual(current_nba_season_start_year(date(2026, 5, 7)), 2025)
        self.assertEqual(nba_season_display_label(2025), "2025-26")

    def test_fall_2026_is_2026_27_nba_season(self) -> None:
        self.assertEqual(current_nba_season_start_year(date(2026, 10, 1)), 2026)
        self.assertEqual(nba_season_display_label(2026), "2026-27")

    def test_assign_dataset_split_uses_fixed_chronological_seasons(self) -> None:
        frame = pd.DataFrame(
            {
                "game_id": ["a", "b", "c"],
                "season": [2023, 2024, 2025],
            }
        )

        split = assign_dataset_split(frame)

        self.assertEqual(split["dataset_split"].tolist(), ["train", "validation", "test"])


if __name__ == "__main__":
    unittest.main()
