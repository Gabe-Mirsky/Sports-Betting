from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.dashboard import build_dashboard_html  # noqa: E402


class TestDashboard(unittest.TestCase):
    def test_dashboard_html_contains_core_tabs(self) -> None:
        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")
        self.assertIn("NBA Kalshi Predictor Dashboard", html)
        self.assertIn('data-tab="overview"', html)
        self.assertIn('data-tab="forward"', html)
        self.assertIn('data-tab="model"', html)
        self.assertIn('data-tab="backtest"', html)
        self.assertIn('data-tab="quality"', html)
        self.assertIn("Probability Buckets", html)
        self.assertIn("Kalshi Coverage by Month", html)
        self.assertIn("Market-Aware Probability Comparison", html)
        self.assertIn("Unmatched Market Gap Reasons", html)
        self.assertIn("Broad NBA Market Discovery", html)
        self.assertIn("Market Line Extraction Audit", html)
        self.assertIn("Multivariate NBA Leg Inventory", html)
        self.assertIn("Security Audit", html)
        self.assertIn("Spread MAE", html)
        self.assertIn("Spread and Total Calibration", html)
        self.assertIn("Optimized Individual Slate", html)
        self.assertIn("Calibrated Individual Slate", html)
        self.assertIn("Market-Blend Calibrated Slate", html)
        self.assertIn("Consensus Calibrated Slate", html)
        self.assertIn("Robust Consensus Slate", html)
        self.assertIn("Signal Stability By Month", html)
        self.assertIn("Daily Slate Risk", html)
        self.assertIn("Edge Calibration", html)
        self.assertIn("Negative Raw-Edge Calibrated Signals", html)
        self.assertIn("Walk-Forward Model Tuning", html)
        self.assertIn("Home-Win Ensemble", html)
        self.assertIn("Fixed-Weight Ensemble Audit", html)
        self.assertIn("Tuned Bankroll", html)
        self.assertIn("Portfolio Comparison", html)
        self.assertIn("Headline Slate Backtest", html)
        self.assertIn("Home-win ensemble backtest", html)
        self.assertIn("Strategy Readiness", html)
        self.assertIn("Signal Rule Sweep", html)
        self.assertIn("Best Rule Monthly Stability", html)
        self.assertIn("Walk-Forward Rule Validation", html)
        self.assertIn("Parlay Correlation Research", html)
        self.assertIn("Forward Recommendations", html)
        self.assertIn("Largest Paper P/L", html)
        self.assertIn("Research and paper trading only", html)


if __name__ == "__main__":
    unittest.main()
