from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.edge_calibration import (  # noqa: E402
    add_expanding_edge_calibration,
    add_expanding_price_aware_edge_calibration,
    audit_calibrated_edges,
    edge_bin_summary,
    price_aware_bin_summary,
    sweep_price_aware_calibration,
)


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
        self.assertAlmostEqual(float(summary.loc[0, "observed_win_rate"]), 0.75)
        self.assertAlmostEqual(float(summary.loc[0, "observed_yes_rate"]), 0.75)
        self.assertAlmostEqual(float(summary.loc[0, "avg_realized_profit_per_share"]), 0.25)

    def test_edge_calibration_is_side_aware_for_no_bets(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "side": "NO",
                    "model_yes_prob": 0.30,
                    "model_prob": 0.70,
                    "market_prob": 0.50,
                    "edge": 0.20,
                    "price_cents": 50,
                    "actual_yes_win": False,
                },
                {
                    "date": "2025-01-02",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "side": "NO",
                    "model_yes_prob": 0.30,
                    "model_prob": 0.70,
                    "market_prob": 0.50,
                    "edge": 0.20,
                    "price_cents": 50,
                    "actual_yes_win": True,
                },
            ]
        )

        summary = edge_bin_summary(rows, edge_bins=[0.0, 0.3])

        self.assertAlmostEqual(float(summary.loc[0, "observed_win_rate"]), 0.5)
        self.assertAlmostEqual(float(summary.loc[0, "avg_model_prob"]), 0.70)
        self.assertAlmostEqual(float(summary.loc[0, "avg_realized_profit_per_share"]), 0.0)

    def test_edge_calibration_uses_candidate_side_when_trade_side_is_blank(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2025-01-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "side": "",
                    "candidate_side": "NO",
                    "model_yes_prob": 0.30,
                    "model_prob": 0.70,
                    "market_prob": 0.50,
                    "edge": 0.20,
                    "price_cents": 50,
                    "actual_yes_win": False,
                }
            ]
        )

        summary = edge_bin_summary(rows, edge_bins=[0.0, 0.3])

        self.assertAlmostEqual(float(summary.loc[0, "observed_win_rate"]), 1.0)
        self.assertAlmostEqual(float(summary.loc[0, "avg_realized_profit_per_share"]), 0.5)

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

    def test_price_aware_calibration_requires_price_bucket_history(self) -> None:
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "date": f"2025-01-{idx + 1:02d}",
                    "game_id": f"g{idx}",
                    "market_ticker": f"M{idx}",
                    "model_yes_prob": 0.45,
                    "market_prob": 0.20,
                    "edge": 0.25,
                    "price_cents": 20,
                    "actual_yes_win": idx < 4,
                }
            )
        frame = pd.DataFrame(rows)

        calibrated, summary = add_expanding_price_aware_edge_calibration(
            frame,
            edge_bins=[0.0, 0.3],
            price_bins=[0, 25, 100],
            min_history_rows=3,
            min_price_history_rows=3,
            shrinkage_rows=1,
        )

        self.assertEqual(summary["rows"], 8)
        self.assertIn("price_bin", calibrated.columns)
        self.assertFalse(bool(calibrated.loc[0, "calibrated_trade"]))
        self.assertTrue(calibrated["calibrated_trade"].any())
        bins = price_aware_bin_summary(calibrated)
        self.assertFalse(bins.empty)
        self.assertIn("calibration_error", bins.columns)

    def test_price_aware_calibration_sweep_returns_ranked_rules(self) -> None:
        rules, summary = sweep_price_aware_calibration(
            self._rows(),
            min_history_options=[1],
            min_price_history_options=[1],
            shrinkage_options=[1],
            min_profit_options=[0.0],
        )

        self.assertEqual(summary["rules_tested"], 1)
        self.assertFalse(rules.empty)
        self.assertIn("score", rules.columns)


if __name__ == "__main__":
    unittest.main()
