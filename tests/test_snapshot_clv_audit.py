from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from strategy.snapshot_clv_audit import build_snapshot_clv_audit  # noqa: E402


class TestSnapshotClvAudit(unittest.TestCase):
    def test_snapshot_audit_reports_distribution_and_concentration(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "trade": True,
                    "side": "YES",
                    "price_cents": 40,
                    "edge": 0.06,
                    "volume": 100,
                    "profit": 0.25,
                    "clv_cents": 4,
                    "snapshot_target": "pregame_best_le_120m",
                    "clv_reference_snapshot": "pregame_5m",
                },
                {
                    "date": "2026-01-02",
                    "trade": True,
                    "side": "NO",
                    "price_cents": 25,
                    "edge": 0.08,
                    "volume": 1000,
                    "profit": -0.25,
                    "clv_cents": -1,
                    "snapshot_target": "pregame_best_le_120m",
                    "clv_reference_snapshot": "pregame_5m",
                },
                {
                    "date": "2026-01-03",
                    "trade": True,
                    "side": "YES",
                    "price_cents": 30,
                    "edge": 0.04,
                    "volume": 50,
                    "profit": 0.0,
                    "clv_cents": 0,
                    "snapshot_target": "pregame_best_le_120m",
                    "clv_reference_snapshot": "pregame_5m",
                },
                {
                    "date": "2026-01-04",
                    "trade": False,
                    "side": "YES",
                    "price_cents": 30,
                    "edge": 0.04,
                    "volume": 50,
                    "profit": 1.0,
                    "clv_cents": 10,
                    "snapshot_target": "pregame_best_le_120m",
                    "clv_reference_snapshot": "pregame_5m",
                },
            ]
        )

        reports, summary = build_snapshot_clv_audit(rows, concentration_top_n=2)

        self.assertEqual(summary["signals"], 3)
        self.assertAlmostEqual(summary["positive_clv_rate"], 1 / 3)
        self.assertIn("by_side", reports)
        self.assertIn("by_side_reference_snapshot", reports)
        self.assertIn("top_positive_clv", reports)
        self.assertGreaterEqual(summary["top_positive_clv_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
