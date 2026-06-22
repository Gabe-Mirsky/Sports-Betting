from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.recommendation_grading_audit import (  # noqa: E402
    _leakage_columns,
    add_clv_math_checks,
    add_profit_math_checks,
    audit_parlays,
    build_recommendation_grading_audit,
)


class TestRecommendationGradingAudit(unittest.TestCase):
    def test_duplicate_match_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recommendations = pd.DataFrame(
                [
                    {"game_id": "g1", "market_ticker": "m1", "recommendation_tier": "research_lean"},
                    {"game_id": "g1", "market_ticker": "m1", "recommendation_tier": "paper_trade_candidate"},
                ]
            )
            results = pd.DataFrame(
                [
                    {"game_id": "g1", "market_ticker": "m1", "actual_yes_win": True},
                    {"game_id": "g1", "market_ticker": "m1", "actual_yes_win": True},
                ]
            )
            graded = pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "market_ticker": "m1",
                        "result_status": "graded",
                        "graded_side": "YES",
                        "graded_price": 60,
                        "actual_yes_win": True,
                        "profit_loss": 0.4,
                    }
                ]
            )
            rec_path, results_path, graded_path = root / "rec.csv", root / "res.csv", root / "graded.csv"
            recommendations.to_csv(rec_path, index=False)
            results.to_csv(results_path, index=False)
            graded.to_csv(graded_path, index=False)

            summary = build_recommendation_grading_audit(rec_path, results_path, graded_path, root / "snaps", root)

            self.assertEqual(summary["match_quality"]["duplicate_recommendation_match_keys"], 1)
            self.assertEqual(summary["match_quality"]["duplicate_result_match_keys"], 1)
            self.assertEqual(summary["match_quality"]["recommendations_matching_multiple_results"], 1)
            self.assertEqual(summary["match_quality"]["multiple_recommendations_matching_one_result"], 1)

    def test_yes_profit_math(self) -> None:
        rows = pd.DataFrame(
            [
                {"graded_side": "YES", "actual_yes_win": True, "graded_price": 60, "profit_loss": 0.4, "result_status": "graded"},
                {"graded_side": "YES", "actual_yes_win": False, "graded_price": 60, "profit_loss": -0.6, "result_status": "graded"},
            ]
        )

        checked = add_profit_math_checks(rows)

        self.assertTrue(checked["audit_profit_math_ok"].all())
        self.assertTrue(checked["audit_winning_profit_ok"].iloc[0])
        self.assertTrue(checked["audit_losing_loss_ok"].iloc[1])

    def test_no_profit_math(self) -> None:
        rows = pd.DataFrame(
            [
                {"graded_side": "NO", "actual_yes_win": False, "graded_price": 35, "profit_loss": 0.65, "result_status": "graded"},
                {"graded_side": "NO", "actual_yes_win": True, "graded_price": 35, "profit_loss": -0.35, "result_status": "graded"},
            ]
        )

        checked = add_profit_math_checks(rows)

        self.assertTrue(checked["audit_profit_math_ok"].all())
        self.assertTrue(checked["audit_winning_profit_ok"].iloc[0])
        self.assertTrue(checked["audit_losing_loss_ok"].iloc[1])

    def test_clv_direction(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "graded_side": "YES",
                    "graded_price": 50,
                    "clv_reference_price_cents": 55,
                    "clv": 5,
                },
                {
                    "graded_side": "NO",
                    "graded_price": 40,
                    "clv_reference_no_price_cents": 35,
                    "clv": 5,
                },
            ]
        )

        checked = add_clv_math_checks(rows)

        self.assertFalse(bool(checked.loc[0, "audit_clv_direction_issue"]))
        self.assertTrue(bool(checked.loc[1, "audit_clv_direction_issue"]))
        self.assertTrue(bool(checked.loc[1, "audit_clv_math_ok"]) is False)

    def test_unmatched_rows_and_summary_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recommendations = pd.DataFrame(
                [
                    {"game_id": "g1", "market_ticker": "m1", "recommendation_tier": "research_lean"},
                    {"game_id": "g2", "market_ticker": "m2", "recommendation_tier": "paper_trade_candidate"},
                ]
            )
            results = pd.DataFrame([{"game_id": "g1", "market_ticker": "m1", "actual_yes_win": True}])
            graded = pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "market_ticker": "m1",
                        "result_status": "graded",
                        "graded_side": "YES",
                        "graded_price": 50,
                        "actual_yes_win": True,
                        "profit_loss": 0.5,
                    },
                    {
                        "game_id": "g2",
                        "market_ticker": "m2",
                        "result_status": "unmatched",
                        "graded_side": "NO",
                        "graded_price": 40,
                        "profit_loss": "",
                    },
                ]
            )
            rec_path, results_path, graded_path = root / "rec.csv", root / "res.csv", root / "graded.csv"
            recommendations.to_csv(rec_path, index=False)
            results.to_csv(results_path, index=False)
            graded.to_csv(graded_path, index=False)

            summary = build_recommendation_grading_audit(rec_path, results_path, graded_path, root / "snaps", root)

            self.assertEqual(summary["match_quality"]["unmatched"], 1)
            self.assertTrue(summary["payout_profit_math"]["total_profit_equals_row_sum"])
            self.assertAlmostEqual(summary["payout_profit_math"]["total_profit"], 0.5)

    def test_leakage_column_detection(self) -> None:
        clean = pd.DataFrame(columns=["game_id", "market_ticker", "model_prob", "edge", "recommendation_tier"])
        leaky = pd.DataFrame(columns=["game_id", "actual_yes_win", "final_score", "realized_profit", "edge"])

        self.assertEqual(_leakage_columns(clean), [])
        self.assertCountEqual(
            _leakage_columns(leaky),
            ["actual_yes_win", "final_score", "realized_profit"],
        )

    def test_parlay_won_lost_grading(self) -> None:
        graded_singles = pd.DataFrame(
            [
                {"market": "BOS at NYK", "graded_side": "YES", "result_status": "graded", "won": True},
                {"market": "LAL at DEN", "graded_side": "NO", "result_status": "graded", "won": False},
                {"market": "MIA at ORL", "graded_side": "YES", "result_status": "graded", "won": True},
            ]
        )
        parlays = pd.DataFrame(
            [
                {
                    "parlay_tier": "research_parlay",
                    "number_of_legs": 2,
                    "correlation_risk": "unknown",
                    "legs": "BOS at NYK YES @ 52.0c | LAL at DEN NO @ 45.0c",
                    "legs_won": 1,
                    "legs_lost": 1,
                    "parlay_won": False,
                },
                {
                    "parlay_tier": "research_parlay",
                    "number_of_legs": 2,
                    "correlation_risk": "unknown",
                    "legs": "BOS at NYK YES @ 52.0c | MIA at ORL YES @ 40.0c",
                    "legs_won": 2,
                    "legs_lost": 0,
                    "parlay_won": True,
                },
            ]
        )

        audit = audit_parlays(parlays, graded_singles)

        self.assertEqual(audit.loc[0, "recomputed_legs_won"], 1)
        self.assertEqual(audit.loc[0, "recomputed_legs_lost"], 1)
        self.assertFalse(bool(audit.loc[0, "recomputed_parlay_won"]))
        self.assertTrue(bool(audit.loc[1, "recomputed_parlay_won"]))
        self.assertTrue(bool(audit["audit_legs_match"].all()))
        self.assertTrue(bool(audit["audit_parlay_won_ok"].all()))


if __name__ == "__main__":
    unittest.main()
