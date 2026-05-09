from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.kalshi_client import (  # noqa: E402
    build_market_entry_template,
    load_mock_kalshi_markets,
    match_games_to_markets,
    public_kalshi_markets_to_game_rows,
    validate_kalshi_markets,
)


class TestKalshiMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = pd.DataFrame(
            [
                {
                    "game_id": "0022400061",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "model_home_win_prob": 0.65,
                    "model_away_win_prob": 0.35,
                    "actual_home_win": 1,
                }
            ]
        )

    def test_yes_home_team_uses_home_probability(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT1",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "yes_mid_cents": 55,
                    "settlement": "YES",
                }
            ]
        )
        matched = match_games_to_markets(self.predictions, markets)
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(matched.loc[0, "model_yes_prob"], 0.65)
        self.assertTrue(bool(matched.loc[0, "actual_yes_win"]))

    def test_yes_away_team_uses_away_probability(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT2",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "NYK",
                    "yes_mid_cents": 45,
                    "settlement": "NO",
                }
            ]
        )
        matched = match_games_to_markets(self.predictions, markets)
        self.assertAlmostEqual(matched.loc[0, "model_yes_prob"], 0.35)
        self.assertFalse(bool(matched.loc[0, "actual_yes_win"]))

    def test_reversed_market_team_order_still_matches(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT3",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "NYK",
                    "away_team_abbr": "BOS",
                    "yes_team_abbr": "BOS",
                    "yes_mid_cents": 55,
                    "settlement": "",
                }
            ]
        )
        matched = match_games_to_markets(self.predictions, markets)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.loc[0, "home_team_abbr"], "BOS")
        self.assertEqual(matched.loc[0, "away_team_abbr"], "NYK")
        self.assertAlmostEqual(matched.loc[0, "model_yes_prob"], 0.65)
        self.assertTrue(bool(matched.loc[0, "actual_yes_win"]))

    def test_common_team_aliases_match_predictions(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT_ALIAS",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "Celtics",
                    "away_team_abbr": "NY",
                    "yes_team_abbr": "Boston Celtics",
                    "yes_mid_cents": 55,
                    "settlement": "YES",
                }
            ]
        )
        matched = match_games_to_markets(self.predictions, markets)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.loc[0, "home_team_abbr"], "BOS")
        self.assertEqual(matched.loc[0, "away_team_abbr"], "NYK")
        self.assertEqual(matched.loc[0, "yes_team_abbr"], "BOS")
        self.assertAlmostEqual(matched.loc[0, "model_yes_prob"], 0.65)

    def test_validation_reports_unmatched_and_bad_price(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "MKT_BAD",
                    "game_date": "2024-10-22",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "yes_mid_cents": "",
                    "settlement": "",
                },
                {
                    "market_ticker": "MKT_UNMATCHED",
                    "game_date": "2024-10-23",
                    "home_team_abbr": "LAL",
                    "away_team_abbr": "GSW",
                    "yes_team_abbr": "LAL",
                    "yes_mid_cents": 50,
                    "settlement": "",
                },
            ]
        )
        report = validate_kalshi_markets(markets, self.predictions)
        self.assertEqual(report["invalid_price_rows"], [0])
        self.assertEqual(report["matched_rows"], 1)
        self.assertEqual(report["unmatched_rows"], 1)

    def test_template_export_uses_prediction_rows(self) -> None:
        output_path = PROJECT_ROOT / "data" / "reports" / "_test_market_template.csv"
        try:
            template = build_market_entry_template(
                self.predictions,
                output_path=output_path,
                start_date="2024-10-22",
                end_date="2024-10-22",
                yes_side="both",
            )
            self.assertEqual(len(template), 2)
            self.assertTrue(output_path.exists())
            self.assertIn("yes_mid_cents", template.columns)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_loader_tags_close_price_fallback(self) -> None:
        path = PROJECT_ROOT / "data" / "reports" / "_test_market_prices.csv"
        try:
            pd.DataFrame(
                [
                    {
                        "market_ticker": "MKT_CLOSE",
                        "game_date": "2024-10-22",
                        "home_team_abbr": "BOS",
                        "away_team_abbr": "NYK",
                        "yes_team_abbr": "BOS",
                        "yes_mid_cents": "",
                        "yes_bid_cents": "",
                        "yes_ask_cents": "",
                        "close_price_cents": 62,
                    }
                ]
            ).to_csv(path, index=False)
            markets = load_mock_kalshi_markets(path)
            self.assertEqual(markets.loc[0, "price_source"], "close_price")
            self.assertAlmostEqual(markets.loc[0, "yes_mid_cents"], 62)
        finally:
            if path.exists():
                path.unlink()

    def test_public_kalshi_game_market_parsing(self) -> None:
        public_markets = [
            {
                "ticker": "KXNBAGAME-26MAY11DETCLE-DET",
                "event_ticker": "KXNBAGAME-26MAY11DETCLE",
                "title": "Game 4: Detroit at Cleveland Winner?",
                "status": "active",
                "yes_bid_dollars": "0.4500",
                "yes_ask_dollars": "0.4700",
                "last_price_dollars": "0.4600",
                "result": "",
                "volume_fp": "3565.23",
                "open_time": "2026-05-04T15:06:00Z",
                "close_time": "2026-05-26T00:00:00Z",
                "updated_time": "2026-05-04T15:06:00Z",
            }
        ]

        markets = public_kalshi_markets_to_game_rows(public_markets)

        self.assertEqual(markets["game_date"].iloc[0].date().isoformat(), "2026-05-11")
        self.assertEqual(markets["away_team_abbr"].iloc[0], "DET")
        self.assertEqual(markets["home_team_abbr"].iloc[0], "CLE")
        self.assertEqual(markets["yes_team_abbr"].iloc[0], "DET")
        self.assertAlmostEqual(markets["yes_bid_cents"].iloc[0], 45.0)
        self.assertAlmostEqual(markets["yes_ask_cents"].iloc[0], 47.0)
        self.assertAlmostEqual(markets["yes_mid_cents"].iloc[0], 46.0)
        self.assertEqual(markets["price_source"].iloc[0], "public_bid_ask_mid")


if __name__ == "__main__":
    unittest.main()
