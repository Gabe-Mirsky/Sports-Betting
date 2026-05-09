from __future__ import annotations

import sys
import shutil
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.coverage import build_kalshi_gap_report  # noqa: E402


class TestKalshiCoverage(unittest.TestCase):
    def test_gap_report_excludes_auto_matches_and_labels_preseason(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_coverage_root"
        if root.exists():
            shutil.rmtree(root)
        try:
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "game_date": "2025-10-10",
                        "home_team_abbr": "TOR",
                        "away_team_abbr": "BOS",
                        "market_ticker": "KXNBAGAME-25OCT10BOSTOR-BOS",
                        "market_title": "Boston vs Toronto Winner?",
                        "yes_team_abbr": "BOS",
                        "status": "finalized",
                        "result": "yes",
                    },
                    {
                        "game_date": "2026-05-05",
                        "home_team_abbr": "OKC",
                        "away_team_abbr": "LAL",
                        "market_ticker": "KXNBAGAME-26MAY05LALOKC-OKC",
                        "market_title": "Game 1: Los Angeles L at Oklahoma City Winner?",
                        "yes_team_abbr": "OKC",
                        "status": "finalized",
                        "result": "yes",
                    },
                ]
            ).to_csv(processed / "kalshi_possible_nba_markets.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "game_date": "2026-05-05",
                        "home_team_abbr": "OKC",
                        "away_team_abbr": "LAL",
                        "match_status": "auto_matched",
                    }
                ]
            ).to_csv(processed / "kalshi_game_market_matches.csv", index=False)

            gap_report = build_kalshi_gap_report(root)
        finally:
            if root.exists():
                shutil.rmtree(root)

        self.assertEqual(len(gap_report), 1)
        self.assertEqual(gap_report["market_ticker"].iloc[0], "KXNBAGAME-25OCT10BOSTOR-BOS")
        self.assertEqual(gap_report["gap_reason"].iloc[0], "preseason_market")


if __name__ == "__main__":
    unittest.main()
