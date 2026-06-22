from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.sportsbook_odds import normalize_sportsbook_odds  # noqa: E402
from strategy.backtest import (  # noqa: E402
    prepare_candlestick_backtest_markets,
    prepare_sportsbook_backtest_markets,
    run_backtest,
    summarize_backtest,
)


class TestBacktest(unittest.TestCase):
    def test_prepare_sportsbook_backtest_uses_no_vig_market_probability(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "season": 2023,
                    "season_type": "Regular Season",
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "model_home_win_prob": 0.58,
                    "model_away_win_prob": 0.42,
                    "actual_home_win": 1,
                }
            ]
        )
        odds = normalize_sportsbook_odds(
            pd.DataFrame(
                [
                    {
                        "game_date": "2024-01-01",
                        "home_team": "BOS",
                        "away_team": "NYK",
                        "home_moneyline": -110,
                        "away_moneyline": 110,
                    }
                ]
            )
        )

        markets, diagnostics = prepare_sportsbook_backtest_markets(predictions, odds)
        trades = run_backtest(markets, starting_bankroll=100, edge_threshold=0.02)

        self.assertEqual(diagnostics["mode"], "sportsbook_market_proxy")
        self.assertEqual(markets.loc[0, "market_source"], "sportsbook")
        self.assertAlmostEqual(float(markets.loc[0, "yes_mid_cents"]), float(markets.loc[0, "home_no_vig_prob"]) * 100)
        self.assertTrue(bool(trades.loc[0, "trade"]))

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
                    "clv_reference_price_cents": 54,
                    "clv_reference_snapshot": "pregame_5m",
                    "clv_cents": 4,
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
        self.assertAlmostEqual(trades.loc[0, "clv_cents"], 4)

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
        self.assertEqual(summary["num_yes_trades"], 0)
        self.assertEqual(summary["num_no_trades"], 0)
        self.assertAlmostEqual(summary["ending_bankroll"], 100)
        self.assertEqual(summary["market_timeline"], "2024-01-01")
        self.assertEqual(summary["trade_timeline"], "n/a")

    def test_backtest_can_buy_no_side(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2024-01-01",
                    "game_id": "g1",
                    "market_ticker": "TEST",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "model_yes_prob": 0.30,
                    "yes_bid": 78,
                    "yes_ask": 80,
                    "actual_yes_win": False,
                    "clv_reference_no_price_cents": 25,
                    "clv_reference_no_snapshot": "pregame_5m",
                }
            ]
        )
        trades = run_backtest(
            markets,
            starting_bankroll=100,
            edge_threshold=0.05,
            max_bet_fraction=0.01,
            allow_no_trades=True,
        )

        self.assertTrue(trades.loc[0, "trade"])
        self.assertEqual(trades.loc[0, "side"], "NO")
        self.assertEqual(trades.loc[0, "candidate_side"], "NO")
        self.assertEqual(trades.loc[0, "price_cents"], 22)
        self.assertGreater(trades.loc[0, "profit"], 0)
        self.assertEqual(trades.loc[0, "clv_cents"], 3)
        summary = summarize_backtest(trades, starting_bankroll=100)
        self.assertEqual(summary["num_no_trades"], 1)
        self.assertEqual(summary["num_yes_trades"], 0)
        self.assertAlmostEqual(summary["no_win_rate"], 1.0)
        self.assertAlmostEqual(summary["no_average_clv_cents"], 3.0)

    def test_untraded_no_candidate_uses_no_side_clv(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2024-01-01",
                    "game_id": "g1",
                    "market_ticker": "TEST",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "model_yes_prob": 0.40,
                    "yes_bid": 89,
                    "yes_ask": 91,
                    "actual_yes_win": True,
                    "clv_reference_price_cents": 91,
                    "clv_reference_snapshot": "pregame_5m",
                    "clv_reference_no_price_cents": 10,
                    "clv_reference_no_snapshot": "pregame_5m",
                }
            ]
        )
        trades = run_backtest(
            markets,
            starting_bankroll=100,
            edge_threshold=0.50,
            max_bet_fraction=0.01,
            allow_no_trades=True,
        )

        self.assertFalse(bool(trades.loc[0, "trade"]))
        self.assertEqual(trades.loc[0, "side"], "")
        self.assertEqual(trades.loc[0, "candidate_side"], "NO")
        self.assertEqual(trades.loc[0, "price_cents"], 11)
        self.assertEqual(trades.loc[0, "clv_reference_price_cents"], 10)
        self.assertEqual(trades.loc[0, "clv_cents"], -1)

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
                    "game_id": "g1",
                    "market_ticker": "MKT1",
                    "series_ticker": "KXNBAGAME",
                    "snapshot_target": "pregame_5m",
                    "snapshot_ts": 2,
                    "yes_bid": 55,
                    "yes_ask": 57,
                    "yes_price": 57,
                    "volume": 30,
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
            max_bid_ask_spread_cents=10,
        )

        self.assertEqual(matched["game_id"].tolist(), ["g1"])
        self.assertEqual(diagnostics["price_rows_after_quality_filter"], 2)
        self.assertEqual(diagnostics["price_rows_after_spread_filter"], 2)
        self.assertEqual(diagnostics["games_with_usable_pregame_price"], 1)
        self.assertEqual(float(matched["pregame_price_60m"].iloc[0]), 51)
        self.assertEqual(float(matched["pregame_price_5m"].iloc[0]), 57)
        self.assertEqual(float(matched["clv_cents"].iloc[0]), 6)

    def test_prepare_candlestick_backtest_filters_wide_spreads(self) -> None:
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
                }
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
                }
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
                    "yes_bid": 40,
                    "yes_ask": 60,
                    "yes_price": 60,
                    "volume": 25,
                    "period_interval": 1,
                    "price_quality": "bid_ask_available",
                }
            ]
        )

        matched, diagnostics = prepare_candlestick_backtest_markets(
            predictions,
            matches,
            prices,
            allowed_price_qualities=("bid_ask_available",),
            require_bid_ask=True,
            max_bid_ask_spread_cents=10,
        )

        self.assertTrue(matched.empty)
        self.assertEqual(diagnostics["price_rows_after_bid_ask_filter"], 1)
        self.assertEqual(diagnostics["price_rows_after_spread_filter"], 0)

    def test_prepare_candlestick_backtest_can_prefer_best_two_hour_snapshot(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "game_date": "2026-01-01",
                    "game_id": "1",
                    "home_team_abbr": "NYK",
                    "away_team_abbr": "BOS",
                    "model_home_win_prob": 0.60,
                    "model_away_win_prob": 0.40,
                    "home_win": True,
                }
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "game_date": "2026-01-01",
                    "game_id": "1",
                    "home_team_abbr": "NYK",
                    "away_team_abbr": "BOS",
                    "market_ticker": "MKT",
                    "series_ticker": "KXNBAGAME",
                    "yes_team_abbr": "NYK",
                    "match_status": "auto_matched",
                }
            ]
        )
        prices = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "market_ticker": "MKT",
                    "series_ticker": "KXNBAGAME",
                    "snapshot_target": "pregame_60m",
                    "snapshot_ts": 100,
                    "yes_price": 55,
                    "yes_bid": 54,
                    "yes_ask": 55,
                    "price_quality": "bid_ask_available",
                    "period_interval": 1,
                    "volume": 100,
                },
                {
                    "game_id": "1",
                    "market_ticker": "MKT",
                    "series_ticker": "KXNBAGAME",
                    "snapshot_target": "pregame_best_le_120m",
                    "snapshot_ts": 200,
                    "yes_price": 50,
                    "yes_bid": 49,
                    "yes_ask": 50,
                    "price_quality": "bid_ask_available",
                    "period_interval": 1,
                    "volume": 100,
                },
            ]
        )

        matched, diagnostics = prepare_candlestick_backtest_markets(
            predictions,
            matches,
            prices,
            preferred_snapshot_targets=["pregame_best_le_120m", "pregame_60m"],
        )

        self.assertEqual(matched["snapshot_target"].iloc[0], "pregame_best_le_120m")
        self.assertEqual(float(matched["yes_mid_cents"].iloc[0]), 50)
        self.assertEqual(diagnostics["preferred_snapshot_targets"][0], "pregame_best_le_120m")


if __name__ == "__main__":
    unittest.main()
