from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.consensus import build_consensus_calibrated_edges  # noqa: E402


class TestConsensus(unittest.TestCase):
    def test_consensus_requires_raw_and_blend_flags(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "calibrated_trade": True,
                    "calibrated_expected_roi": 0.10,
                    "actual_yes_win": True,
                    "realized_profit_per_share": 0.40,
                },
                {
                    "date": "2025-01-02",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "calibrated_trade": True,
                    "calibrated_expected_roi": 0.20,
                    "actual_yes_win": False,
                    "realized_profit_per_share": -0.50,
                },
            ]
        )
        blend = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "calibrated_trade": True,
                    "calibrated_expected_roi": 0.08,
                },
                {
                    "date": "2025-01-02",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "calibrated_trade": False,
                    "calibrated_expected_roi": 0.15,
                },
            ]
        )

        consensus, summary = build_consensus_calibrated_edges(raw, blend)

        self.assertEqual(consensus["consensus_trade"].tolist(), [True, False])
        self.assertAlmostEqual(float(consensus.loc[0, "consensus_expected_roi"]), 0.08)
        self.assertEqual(summary["consensus_trades"], 1)
        self.assertEqual(summary["trade_timeline"], "2025-01-01")


if __name__ == "__main__":
    unittest.main()
