from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.backtest import prepare_candlestick_backtest_markets, run_backtest, summarize_backtest  # noqa: E402


class TestBacktest(unittest.TestCase):
    def test_bankroll_updates_after_winning_yes_trade(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2024-01-01",
                    "game_id": "g1",
                    "market_ticker": "TEST",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "model_yes_prob": 0.70,
                    "yes_mid_cents": 50,
                    "actual_yes_win": True,
                }
            ]
        )
        trades = run_backtest(
            markets,
            starting_bankroll=100,
            edge_threshold=0.05,
            max_bet_fraction=0.01,
        )
        self.assertTrue(trades.loc[0, "trade"])
        self.assertEqual(trades.loc[0, "shares"], 2)
        self.assertAlmostEqual(trades.loc[0, "cost"], 1.0)
        self.assertAlmostEqual(trades.loc[0, "payout"], 2.0)
        self.assertAlmostEqual(trades.loc[0, "bankroll_after"], 101.0)

    def test_summary_handles_no_trades(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2024-01-01",
                    "game_id": "g1",
                    "market_ticker": "TEST",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "model_yes_prob": 0.51,
                    "yes_mid_cents": 50,
                    "actual_yes_win": True,
                }
            ]
        )
        trades = run_backtest(markets, starting_bankroll=100, edge_threshold=0.05)
        summary = summarize_backtest(trades, starting_bankroll=100)
        self.assertEqual(summary["num_trades"], 0)
        self.assertAlmostEqual(summary["ending_bankroll"], 100)
        self.assertEqual(summary["market_timeline"], "2024-01-01")
        self.assertEqual(summary["trade_timeline"], "n/a")

    def test_prepare_candlestick_backtest_filters_weak_prices(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-01-01",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "model_home_win_prob": 0.60,
                    "model_away_win_prob": 0.40,
                    "actual_home_win": 1,
                    "season_type": "Regular Season",
                },
                {
                    "game_id": "g2",
                    "game_date": "2026-01-02",
                    "home_team_abbr": "CCC",
                    "away_team_abbr": "DDD",
                    "model_home_win_prob": 0.55,
                    "model_away_win_prob": 0.45,
                    "actual_home_win": 0,
                    "season_type": "Regular Season",
                },
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-01-01",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "market_ticker": "MKT1",
                    "series_ticker": "KXNBAGAME",
                    "yes_team_abbr": "AAA",
                    "match_status": "auto_matched",
                },
                {
                    "game_id": "g2",
                    "game_date": "2026-01-02",
                    "home_team_abbr": "CCC",
                    "away_team_abbr": "DDD",
                    "market_ticker": "MKT2",
                    "series_ticker": "KXNBAGAME",
                    "yes_team_abbr": "DDD",
                    "match_status": "auto_matched",
                },
            ]
        )
        prices = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "MKT1",
                    "series_ticker": "KXNBAGAME",
                    "snapshot_target": "pregame_60m",
                    "snapshot_ts": 1,
                    "yes_bid": 49,
                    "yes_ask": 51,
                    "yes_price": 51,
                    "volume": 25,
                    "period_interval": 1,
                    "price_quality": "bid_ask_available",
                },
                {
                    "game_id": "g2",
                    "market_ticker": "MKT2",
                    "series_ticker": "KXNBAGAME",
                    "snapshot_target": "pregame_60m",
                    "snapshot_ts": 1,
                    "yes_bid": 45,
                    "yes_ask": 47,
                    "yes_price": 47,
                    "volume": 0,
                    "period_interval": 1440,
                    "price_quality": "daily_candle_low_quality",
                },
            ]
        )

        matched, diagnostics = prepare_candlestick_backtest_markets(
            predictions,
            matches,
            prices,
            min_volume=10,
            allowed_price_qualities=("bid_ask_available",),
            require_bid_ask=True,
            max_candle_interval_minutes=60,
        )

        self.assertEqual(matched["game_id"].tolist(), ["g1"])
        self.assertEqual(diagnostics["price_rows_after_quality_filter"], 1)
        self.assertEqual(diagnostics["games_with_usable_pregame_price"], 1)


if __name__ == "__main__":
    unittest.main()
