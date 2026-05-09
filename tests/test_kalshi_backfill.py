from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.kalshi_backfill import filter_possible_nba_markets  # noqa: E402


class TestKalshiBackfill(unittest.TestCase):
    def test_filter_fills_game_fields_from_kxnbagame_ticker(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-26APR17GSWPHX-PHX",
                    "market_title": "Golden State at Phoenix Winner?",
                    "game_date": "",
                    "home_team_abbr": "",
                    "away_team_abbr": "",
                    "yes_team_abbr": "",
                }
            ]
        )

        possible = filter_possible_nba_markets(markets)

        self.assertEqual(possible.loc[0, "game_date"], "2026-04-17")
        self.assertEqual(possible.loc[0, "away_team_abbr"], "GSW")
        self.assertEqual(possible.loc[0, "home_team_abbr"], "PHX")
        self.assertEqual(possible.loc[0, "yes_team_abbr"], "PHX")
        self.assertEqual(possible.loc[0, "no_team_abbr"], "GSW")

    def test_filter_excludes_other_nba_series_from_full_game_match_pool(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-26APR17GSWPHX-PHX",
                    "series_ticker": "KXNBAGAME",
                    "market_title": "Golden State at Phoenix Winner?",
                },
                {
                    "market_ticker": "KXNBA1HWINNER-26APR17GSWPHX-PHX",
                    "series_ticker": "KXNBA1HWINNER",
                    "market_title": "Golden State at Phoenix: First Half Winner?",
                },
                {
                    "market_ticker": "KXNBAMENTION-26APR17GSWPHX-CURRY",
                    "series_ticker": "KXNBAMENTION",
                    "market_title": "What will the announcers say during Golden State vs Phoenix?",
                },
            ]
        )

        possible = filter_possible_nba_markets(markets)

        self.assertEqual(possible["market_ticker"].tolist(), ["KXNBAGAME-26APR17GSWPHX-PHX"])


if __name__ == "__main__":
    unittest.main()
