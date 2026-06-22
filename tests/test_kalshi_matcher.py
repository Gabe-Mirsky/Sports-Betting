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

    def test_tied_team_markets_prefer_home_yes_side(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "3",
                    "game_date": "2026-03-10",
                    "home_team_abbr": "HOU",
                    "away_team_abbr": "TOR",
                }
            ]
        )
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-26MAR10TORHOU-TOR",
                    "series_ticker": "KXNBAGAME",
                    "event_ticker": "KXNBAGAME-26MAR10TORHOU",
                    "market_title": "Toronto at Houston Winner?",
                    "market_subtitle": "",
                    "close_time": "2026-03-11T00:00:00Z",
                },
                {
                    "market_ticker": "KXNBAGAME-26MAR10TORHOU-HOU",
                    "series_ticker": "KXNBAGAME",
                    "event_ticker": "KXNBAGAME-26MAR10TORHOU",
                    "market_title": "Toronto at Houston Winner?",
                    "market_subtitle": "",
                    "close_time": "2026-03-11T00:00:00Z",
                },
            ]
        )

        matches = match_games_to_kalshi_markets(games, markets)

        self.assertEqual(matches.loc[0, "match_status"], "auto_matched")
        self.assertEqual(matches.loc[0, "yes_team_abbr"], "HOU")
        self.assertEqual(matches.loc[0, "market_ticker"], "KXNBAGAME-26MAR10TORHOU-HOU")

    def test_rejects_reversed_kxnbagame_orientation(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "4",
                    "game_date": "2026-01-15",
                    "home_team_abbr": "MEM",
                    "away_team_abbr": "ORL",
                }
            ]
        )
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-26JAN15MEMORL-MEM",
                    "series_ticker": "KXNBAGAME",
                    "event_ticker": "KXNBAGAME-26JAN15MEMORL",
                    "market_title": "Memphis at Orlando Winner?",
                    "market_subtitle": "",
                    "expected_expiration_time": "2026-01-15T22:00:00Z",
                }
            ]
        )

        matches = match_games_to_kalshi_markets(games, markets)

        self.assertEqual(matches.loc[0, "match_status"], "no_match")
        self.assertIn("does not match", matches.loc[0, "match_notes"])


if __name__ == "__main__":
    unittest.main()
