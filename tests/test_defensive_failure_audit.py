from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.defensive_failure_audit import build_defensive_failure_audit  # noqa: E402


class TestDefensiveFailureAudit(unittest.TestCase):
    def test_defensive_failure_audit_compares_failure_month(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-02-01",
                    "game_id": "g1",
                    "market_ticker": "A",
                    "walk_forward_defensive_signal": True,
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                    "yes_team_abbr": "BOS",
                    "price_cents": 15,
                    "calibrated_expected_roi": 2.0,
                    "edge": 0.05,
                    "volume": 100,
                    "open_interest": 1000,
                    "clv_cents": 4,
                    "realized_profit_per_share": 0.2,
                },
                {
                    "date": "2026-03-01",
                    "game_id": "g2",
                    "market_ticker": "B",
                    "walk_forward_defensive_signal": True,
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "MIA",
                    "yes_team_abbr": "BOS",
                    "price_cents": 15,
                    "calibrated_expected_roi": 2.0,
                    "edge": 0.05,
                    "volume": 100,
                    "open_interest": 1000,
                    "clv_cents": -1,
                    "realized_profit_per_share": -0.1,
                },
            ]
        )

        reports, summary = build_defensive_failure_audit(
            rows,
            failure_month="2026-03",
            min_segment_rows=1,
        )

        self.assertEqual(summary["failure_month"], "2026-03")
        self.assertLess(summary["failure_month_avg_profit_per_share"], 0)
        self.assertIn("failure_vs_other_price_bucket", reports)
        self.assertFalse(reports["failure_month_rows"].empty)
        self.assertEqual(summary["schedule_context_status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
