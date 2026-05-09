from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.kalshi_matcher import match_games_to_kalshi_markets  # noqa: E402


class TestKalshiMatcher(unittest.TestCase):
    def test_high_confidence_team_win_market_auto_matches(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                }
            ]
        )
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-24OCT22NYKBOS-BOS",
                    "series_ticker": "KXNBAGAME",
                    "event_ticker": "KXNBAGAME-24OCT22NYKBOS",
                    "market_title": "Will the Boston Celtics beat the New York Knicks?",
                    "market_subtitle": "",
                    "close_time": "2024-10-22T23:00:00Z",
                }
            ]
        )

        matches = match_games_to_kalshi_markets(games, markets)

        self.assertEqual(matches.loc[0, "match_status"], "auto_matched")
        self.assertEqual(matches.loc[0, "yes_team_abbr"], "BOS")
        self.assertGreaterEqual(matches.loc[0, "match_score"], 0.85)

    def test_props_are_rejected_even_with_team_names(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                }
            ]
        )
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "PROP-1",
                    "market_title": "Will the Boston Celtics score over 110 points vs the New York Knicks?",
                    "market_subtitle": "",
                    "close_time": "2024-10-22T23:00:00Z",
                }
            ]
        )

        matches = match_games_to_kalshi_markets(games, markets)

        self.assertEqual(matches.loc[0, "match_status"], "no_match")

    def test_exact_kalshi_ticker_can_match_odd_title(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "2",
                    "game_date": "2026-05-07",
                    "home_team_abbr": "OKC",
                    "away_team_abbr": "LAL",
                }
            ]
        )
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-26MAY07LALOKC-LAL",
                    "series_ticker": "KXNBAGAME",
                    "event_ticker": "KXNBAGAME-26MAY07LALOKC",
                    "market_title": "Game 2: Los Angeles L at Oklahoma City Winner?",
                    "market_subtitle": "",
                    "expected_expiration_time": "2026-05-08T04:30:00Z",
                }
            ]
        )

        matches = match_games_to_kalshi_markets(games, markets)

        self.assertEqual(matches.loc[0, "match_status"], "auto_matched")
        self.assertEqual(matches.loc[0, "yes_team_abbr"], "LAL")


if __name__ == "__main__":
    unittest.main()
