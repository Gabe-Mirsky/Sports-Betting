from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.threshold_sweep import parse_thresholds, run_threshold_sweep  # noqa: E402


class TestThresholdSweep(unittest.TestCase):
    def test_parse_thresholds_sorts_and_deduplicates(self) -> None:
        self.assertEqual(parse_thresholds("0.05, 0.02, 0.05"), [0.02, 0.05])

    def test_parse_thresholds_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_thresholds(" , ")

    def test_trade_count_decreases_as_threshold_rises(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "game_date": "2024-01-01",
                    "game_id": "g1",
                    "market_ticker": "MKT1",
                    "home_team_abbr": "AAA",
                    "away_team_abbr": "BBB",
                    "yes_team_abbr": "AAA",
                    "model_yes_prob": 0.60,
                    "yes_mid_cents": 50,
                    "actual_yes_win": True,
                },
                {
                    "game_date": "2024-01-02",
                    "game_id": "g2",
                    "market_ticker": "MKT2",
                    "home_team_abbr": "CCC",
                    "away_team_abbr": "DDD",
                    "yes_team_abbr": "CCC",
                    "model_yes_prob": 0.70,
                    "yes_mid_cents": 50,
                    "actual_yes_win": True,
                },
            ]
        )
        sweep = run_threshold_sweep(markets, thresholds=[0.05, 0.15])
        low_count = int(sweep.loc[sweep["edge_threshold"] == 0.05, "num_trades"].iloc[0])
        high_count = int(sweep.loc[sweep["edge_threshold"] == 0.15, "num_trades"].iloc[0])
        self.assertGreater(low_count, high_count)


if __name__ == "__main__":
    unittest.main()
