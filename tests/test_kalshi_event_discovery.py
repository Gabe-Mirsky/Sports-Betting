from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.kalshi_event_discovery import _has_nba_signal, fetch_series_list_candidates  # noqa: E402


class FakeSeriesListClient:
    def get_series_list(self, params: dict) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": "KXNBAGAME",
                    "title": "NBA game winner",
                    "category": "Sports",
                    "tags": ["NBA", "Basketball"],
                },
                {
                    "ticker": "KXNFLGAME",
                    "title": "NFL game winner",
                    "category": "Sports",
                    "tags": ["Football"],
                },
            ]
        )


class TestKalshiEventDiscovery(unittest.TestCase):
    def test_has_nba_signal_from_team_alias(self) -> None:
        self.assertTrue(_has_nba_signal("will the boston celtics win"))
        self.assertFalse(_has_nba_signal("will the green bay packers win"))

    def test_fetch_series_list_candidates_filters_nba_rows(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_event_discovery"
        all_path = root / "series.csv"
        candidates_path = root / "candidates.csv"

        _, candidates, summary = fetch_series_list_candidates(
            client=FakeSeriesListClient(),
            output_path=all_path,
            candidates_path=candidates_path,
        )

        self.assertEqual(candidates["ticker"].tolist(), ["KXNBAGAME"])
        self.assertEqual(summary["candidate_tickers"], ["KXNBAGAME"])
        self.assertTrue(all_path.exists())
        self.assertTrue(candidates_path.exists())


if __name__ == "__main__":
    unittest.main()
