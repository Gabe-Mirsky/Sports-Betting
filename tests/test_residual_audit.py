from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.residual_audit import build_residual_audit, run_residual_guardrail_sweep  # noqa: E402


class TestResidualAudit(unittest.TestCase):
    def test_residual_audit_buckets_calibrated_residuals(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 40,
                    "market_prob": 0.40,
                    "model_prob": 0.50,
                    "edge": 0.10,
                    "calibrated_win_rate": 0.55,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.60,
                    "clv_cents": 1,
                    "volume": 100,
                },
                {
                    "date": "2026-01-02",
                    "calibrated_trade": False,
                    "calibrated_side": "YES",
                    "price_cents": 40,
                    "market_prob": 0.40,
                    "model_prob": 0.50,
                    "edge": 0.10,
                    "calibrated_win_rate": 0.55,
                    "actual_contract_win": False,
                    "realized_profit_per_share": -0.40,
                    "clv_cents": -1,
                    "volume": 100,
                },
            ]
        )

        reports, summary = build_residual_audit(rows)

        self.assertEqual(summary["signals"], 1)
        self.assertAlmostEqual(summary["avg_calibrated_residual"], 0.15)
        self.assertIn("by_side_calibrated_residual", reports)
        self.assertFalse(reports["by_side_calibrated_residual"].empty)

    def test_residual_guardrail_sweep_uses_prior_history(self) -> None:
        rows = []
        for index, date in enumerate(pd.date_range("2026-01-01", periods=8, freq="D")):
            rows.append(
                {
                    "date": date,
                    "calibrated_trade": True,
                    "calibrated_side": "YES",
                    "price_cents": 40,
                    "market_prob": 0.40,
                    "model_prob": 0.50,
                    "edge": 0.10,
                    "calibrated_win_rate": 0.45,
                    "actual_contract_win": True,
                    "realized_profit_per_share": 0.60,
                    "clv_cents": 1,
                    "volume": 100,
                }
            )
        frame = pd.DataFrame(rows)

        rules, selected, summary = run_residual_guardrail_sweep(
            frame,
            min_history_options=[2],
            min_prior_calibration_error_options=[0.0],
            min_prior_positive_clv_rate_options=[0.5],
            min_prior_profit_options=[0.0],
        )

        self.assertEqual(summary["rules_tested"], 1)
        self.assertFalse(rules.empty)
        self.assertGreater(len(selected), 0)


if __name__ == "__main__":
    unittest.main()
