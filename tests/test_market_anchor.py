from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.market_anchor import sweep_market_anchor  # noqa: E402


class TestMarketAnchor(unittest.TestCase):
    def test_market_anchor_recomputes_backtest_edges(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2026-01-01",
                    "game_id": "g1",
                    "market_ticker": "M1",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "model_yes_prob": 0.70,
                    "yes_bid": 49,
                    "yes_ask": 51,
                    "actual_yes_win": True,
                    "clv_reference_price_cents": 55,
                    "clv_reference_snapshot": "pregame_5m",
                    "clv_reference_no_price_cents": 45,
                    "clv_reference_no_snapshot": "pregame_5m",
                },
                {
                    "game_date": "2026-01-02",
                    "game_id": "g2",
                    "market_ticker": "M2",
                    "home_team_abbr": "CCC",
                    "away_team_abbr": "DDD",
                    "yes_team_abbr": "CCC",
                    "model_yes_prob": 0.35,
                    "yes_bid": 60,
                    "yes_ask": 62,
                    "actual_yes_win": False,
                    "clv_reference_price_cents": 60,
                    "clv_reference_snapshot": "pregame_5m",
                    "clv_reference_no_price_cents": 42,
                    "clv_reference_no_snapshot": "pregame_5m",
                },
            ]
        )

        results, summary = sweep_market_anchor(
            markets,
            model_weights=[1.0],
            edge_thresholds=[0.05],
            max_bet_fraction=0.01,
        )

        self.assertFalse(results.empty)
        self.assertEqual(summary["rules_tested"], 1)
        self.assertEqual(int(results.iloc[0]["num_trades"]), 2)
        self.assertIn("best_average_clv_cents", summary)
        self.assertIn("broad_best_average_clv_cents", summary)


if __name__ == "__main__":
    unittest.main()
