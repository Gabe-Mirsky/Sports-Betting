from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.line_market_eval import prepare_line_market_model_eval, summarize_line_market_model_eval  # noqa: E402


class TestLineMarketEval(unittest.TestCase):
    def test_prepare_line_market_model_eval_maps_spread_and_total_probabilities(self) -> None:
        prices = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-03",
                    "home_team_abbr": "CLE",
                    "away_team_abbr": "TOR",
                    "market_category": "spread_handicap",
                    "market_ticker": "KXNBASPREAD-26MAY03TORCLE-CLE8",
                    "line_value": 8.5,
                    "yes_team_abbr": "CLE",
                    "snapshot_target": "pregame_60m",
                    "price_quality": "bid_ask_available",
                    "yes_price": 51.0,
                },
                {
                    "game_date": "2026-05-03",
                    "home_team_abbr": "CLE",
                    "away_team_abbr": "TOR",
                    "market_category": "total_points_over_under",
                    "market_ticker": "KXNBATOTAL-26MAY03TORCLE-211",
                    "line_value": 211.5,
                    "direction": "over",
                    "snapshot_target": "pregame_60m",
                    "price_quality": "bid_ask_available",
                    "yes_price": 62.0,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-03",
                    "home_team_abbr": "CLE",
                    "away_team_abbr": "TOR",
                    "pred_home_margin": 10.0,
                    "pred_total_points": 220.0,
                    "margin_residual_std_train": 12.0,
                    "total_residual_std_train": 18.0,
                    "target_home_margin": 12.0,
                    "target_total_points": 216.0,
                }
            ]
        )

        eval_rows = prepare_line_market_model_eval(prices, predictions, edge_threshold=0.05)
        by_ticker = eval_rows.set_index("market_ticker")

        self.assertEqual(len(eval_rows), 2)
        self.assertGreater(by_ticker.loc["KXNBASPREAD-26MAY03TORCLE-CLE8", "model_yes_prob"], 0.5)
        self.assertTrue(by_ticker.loc["KXNBASPREAD-26MAY03TORCLE-CLE8", "actual_yes"])
        self.assertGreater(by_ticker.loc["KXNBATOTAL-26MAY03TORCLE-211", "model_yes_prob"], 0.5)
        self.assertTrue(by_ticker.loc["KXNBATOTAL-26MAY03TORCLE-211", "actual_yes"])

    def test_summarize_line_market_model_eval_stays_not_ready_with_small_sample(self) -> None:
        eval_rows = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-03",
                    "market_category": "spread_handicap",
                    "actual_yes": True,
                    "model_yes_prob": 0.60,
                    "edge": 0.10,
                    "trade_signal": True,
                    "profit_per_contract": 0.49,
                }
            ]
        )

        summary = summarize_line_market_model_eval(eval_rows)

        self.assertEqual(summary["status"], "not_ready")
        self.assertEqual(summary["signals"], 1)


if __name__ == "__main__":
    unittest.main()
