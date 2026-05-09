from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.edge_calibration import add_expanding_edge_calibration, audit_calibrated_edges, edge_bin_summary  # noqa: E402


class TestEdgeCalibration(unittest.TestCase):
    def _rows(self) -> pd.DataFrame:
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "date": f"2025-01-{idx + 1:02d}",
                    "game_id": f"g{idx}",
                    "market_ticker": f"M{idx}",
                    "model_yes_prob": 0.65,
                    "market_prob": 0.50,
                    "edge": 0.15,
                    "price_cents": 50,
                    "actual_yes_win": idx < 6,
                }
            )
        return pd.DataFrame(rows)

    def test_edge_bin_summary_computes_realized_profit(self) -> None:
        summary = edge_bin_summary(self._rows(), edge_bins=[0.0, 0.2])

        self.assertEqual(int(summary.loc[0, "markets"]), 8)
        self.assertAlmostEqual(float(summary.loc[0, "observed_yes_rate"]), 0.75)
        self.assertAlmostEqual(float(summary.loc[0, "avg_realized_profit_per_share"]), 0.25)

    def test_expanding_calibration_uses_prior_rows_only(self) -> None:
        calibrated, summary = add_expanding_edge_calibration(
            self._rows(),
            edge_bins=[0.0, 0.2],
            min_history_rows=3,
            shrinkage_rows=1,
        )

        self.assertFalse(bool(calibrated.loc[0, "calibrated_trade"]))
        self.assertTrue(calibrated["calibrated_trade"].any())
        self.assertEqual(summary["rows"], 8)

    def test_same_date_rows_do_not_feed_each_other(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "model_yes_prob": 0.65,
                    "market_prob": 0.50,
                    "edge": 0.15,
                    "price_cents": 50,
                    "actual_yes_win": True,
                },
                {
                    "date": "2025-01-01",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "model_yes_prob": 0.65,
                    "market_prob": 0.50,
                    "edge": 0.15,
                    "price_cents": 50,
                    "actual_yes_win": True,
                },
                {
                    "date": "2025-01-02",
                    "game_id": "g3",
                    "market_ticker": "C",
                    "model_yes_prob": 0.65,
                    "market_prob": 0.50,
                    "edge": 0.15,
                    "price_cents": 50,
                    "actual_yes_win": True,
                },
            ]
        )

        calibrated, _ = add_expanding_edge_calibration(
            rows,
            edge_bins=[0.0, 0.2],
            min_history_rows=1,
            shrinkage_rows=1,
        )

        self.assertEqual(calibrated.loc[0, "edge_bin_history_rows"], 0)
        self.assertEqual(calibrated.loc[1, "edge_bin_history_rows"], 0)
        self.assertEqual(calibrated.loc[2, "edge_bin_history_rows"], 2)

    def test_audit_calibrated_edges_counts_negative_raw_edges(self) -> None:
        calibrated, _ = add_expanding_edge_calibration(
            self._rows(),
            edge_bins=[-0.2, 0.0, 0.2],
            min_history_rows=1,
            shrinkage_rows=1,
        )
        calibrated.loc[calibrated.index[-1], "edge"] = -0.03
        calibrated.loc[calibrated.index[-1], "edge_bin"] = "(-0.2, 0.0]"
        calibrated.loc[calibrated.index[-1], "calibrated_trade"] = True

        audit, negative, summary = audit_calibrated_edges(calibrated)

        self.assertFalse(audit.empty)
        self.assertEqual(summary["negative_raw_edge_calibrated_trades"], 1)
        self.assertEqual(len(negative), 1)


if __name__ == "__main__":
    unittest.main()
