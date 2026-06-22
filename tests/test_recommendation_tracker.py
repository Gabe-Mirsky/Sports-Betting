from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.recommendation_tracker import (  # noqa: E402
    build_recommendation_performance_summary,
    grade_parlay_recommendations,
    grade_single_recommendations,
    save_recommendation_snapshot,
)


class TestRecommendationTracker(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        fair = pd.DataFrame(
            [
                {
                    "game_date": "2026-01-01",
                    "game_id": "g1",
                    "market_ticker": "m1",
                    "market": "BOS at NYK",
                    "recommendation_tier": "research_lean",
                    "research_side": "YES",
                    "research_price": 52,
                    "research_model_probability": 0.60,
                    "research_market_implied_probability": 0.52,
                    "edge": 0.08,
                    "confidence_label": "medium",
                },
                {
                    "game_date": "2026-01-01",
                    "game_id": "g2",
                    "market_ticker": "m2",
                    "market": "LAL at DEN",
                    "recommendation_tier": "paper_trade_candidate",
                    "research_side": "NO",
                    "research_price": 45,
                    "research_model_probability": 0.62,
                    "research_market_implied_probability": 0.45,
                    "edge": 0.17,
                    "confidence_label": "high",
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "market_ticker": "m1",
                    "actual_yes_win": True,
                    "clv_reference_price_cents": 57,
                    "clv_reference_no_price_cents": 44,
                },
                {
                    "game_id": "g2",
                    "market_ticker": "m2",
                    "actual_yes_win": True,
                    "clv_reference_price_cents": 62,
                    "clv_reference_no_price_cents": 39,
                },
            ]
        )
        parlay = pd.DataFrame(
            [
                {
                    "parlay_tier": "research_parlay",
                    "research_only": True,
                    "approved": False,
                    "legs": "BOS at NYK YES @ 52.0c edge 0.080 | LAL at DEN NO @ 45.0c edge 0.170",
                    "number_of_legs": 2,
                    "offered_payout": 4.0,
                    "correlation_risk": "unknown",
                }
            ]
        )
        paper = parlay.assign(parlay_tier="paper_parlay")
        fair_path = root / "fair_price_signals.csv"
        research_path = root / "research_parlay_candidates.csv"
        paper_path = root / "paper_parlay_candidates.csv"
        results_path = root / "backtest_trades.csv"
        fair.to_csv(fair_path, index=False)
        parlay.to_csv(research_path, index=False)
        paper.to_csv(paper_path, index=False)
        results.to_csv(results_path, index=False)
        return fair_path, research_path, paper_path, results_path

    def test_save_recommendation_snapshot_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fair, research, paper, _ = self._write_inputs(root)
            first = save_recommendation_snapshot(fair, research, paper, root / "snapshots", run_date="2026-01-02")
            second = save_recommendation_snapshot(fair, research, paper, root / "snapshots", run_date="2026-01-02")

            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
            first_fair = pd.read_csv(first["files_created"]["fair_price_signals"])
            self.assertIn("snapshot_date", first_fair.columns)
            self.assertIn("snapshot_id", first_fair.columns)
            self.assertTrue(Path(first["manifest_path"]).exists())

    def test_grade_single_recommendations_calculates_results_profit_and_clv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fair, _, _, results = self._write_inputs(root)

            graded = grade_single_recommendations(fair, results)

            self.assertEqual(set(graded["result_status"]), {"graded"})
            self.assertTrue(bool(graded.loc[graded["game_id"].eq("g1"), "won"].iloc[0]))
            self.assertFalse(bool(graded.loc[graded["game_id"].eq("g2"), "won"].iloc[0]))
            self.assertAlmostEqual(float(graded.loc[graded["game_id"].eq("g1"), "profit_loss"].iloc[0]), 0.48)
            self.assertAlmostEqual(float(graded.loc[graded["game_id"].eq("g1"), "clv"].iloc[0]), 5.0)
            self.assertAlmostEqual(float(graded.loc[graded["game_id"].eq("g2"), "clv"].iloc[0]), -6.0)

    def test_grade_parlay_recommendations_checks_all_legs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fair, research, _, results = self._write_inputs(root)
            singles = grade_single_recommendations(fair, results)

            graded = grade_parlay_recommendations(research, singles)

            self.assertEqual(graded.loc[0, "legs_won"], 1)
            self.assertEqual(graded.loc[0, "legs_lost"], 1)
            self.assertFalse(bool(graded.loc[0, "parlay_won"]))
            self.assertEqual(graded.loc[0, "combined_result_status"], "graded")
            self.assertAlmostEqual(float(graded.loc[0, "estimated_profit_loss"]), -1.0)
            self.assertIn("lost:LAL at DEN:NO", graded.loc[0, "failed_leg_reason"])

    def test_build_recommendation_performance_summary_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fair, research, paper, results = self._write_inputs(root)
            singles = grade_single_recommendations(fair, results)
            research_parlays = grade_parlay_recommendations(research, singles)
            paper_parlays = grade_parlay_recommendations(paper, singles)

            summary = build_recommendation_performance_summary(singles, research_parlays, paper_parlays, root)

            self.assertEqual(summary["total_research_leans"], 1)
            self.assertEqual(summary["total_paper_candidates"], 1)
            self.assertEqual(summary["graded_rows"], 2)
            self.assertTrue((root / "recommendation_performance_summary.json").exists())
            self.assertTrue((root / "graded_single_recommendations.csv").exists())
            self.assertTrue((root / "graded_research_parlays.csv").exists())
            self.assertTrue((root / "graded_paper_parlays.csv").exists())
            self.assertTrue((root / "recommendation_failure_buckets.csv").exists())


if __name__ == "__main__":
    unittest.main()
