from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy.no_settlement_calibration import (  # noqa: E402
    SuppressionRule,
    apply_prior_suppression_rule,
    break_even_probability,
    build_no_settlement_calibration_audit,
    clv_vs_profit,
    failure_segments,
    prepare_no_settlement_rows,
    summarize_segments,
)


def _trades() -> pd.DataFrame:
    rows = []
    specs = [
        ("2026-01-01", "1", 30, 0.45, False, 2, 2, "AAA", 200),
        ("2026-01-02", "2", 35, 0.50, True, -1, 2, "BBB", 200),
        ("2026-02-01", "3", 40, 0.60, False, 3, 3, "AAA", 50),
        ("2026-02-02", "4", 50, 0.70, True, 4, 3, "CCC", 50),
        ("2026-03-01", "5", 25, 0.35, False, -3, 4, "AAA", 500),
        ("2026-03-02", "6", 45, 0.60, True, 5, 3, "DDD", 500),
    ]
    for date, game_id, price, model_prob, actual_yes_win, clv, shares, team, volume in specs:
        actual_no_win = not actual_yes_win
        cost = shares * price / 100
        payout = shares if actual_no_win else 0
        rows.append(
            {
                "date": date,
                "game_id": game_id,
                "market_ticker": f"T{game_id}",
                "trade": True,
                "side": "NO",
                "candidate_side": "NO",
                "yes_team_abbr": team,
                "price_cents": price,
                "model_prob": model_prob,
                "market_prob": price / 100,
                "edge": model_prob - price / 100,
                "shares": shares,
                "cost": cost,
                "profit": payout - cost,
                "actual_yes_win": actual_yes_win,
                "clv_cents": clv,
                "volume": volume,
                "bankroll_after": 100 + payout - cost,
            }
        )
    return pd.DataFrame(rows)


class TestNoSettlementCalibration(unittest.TestCase):
    def test_no_settlement_logic(self) -> None:
        rows = prepare_no_settlement_rows(_trades())

        self.assertEqual(rows["actual_no_win"].tolist(), [True, False, True, False, True, False])
        self.assertAlmostEqual(rows.loc[0, "profit_per_share"], 0.70)
        self.assertAlmostEqual(rows.loc[1, "profit_per_share"], -0.35)

    def test_break_even_probability_from_no_buy_price(self) -> None:
        self.assertAlmostEqual(break_even_probability(42), 0.42)

    def test_calibration_bucket_math(self) -> None:
        rows = prepare_no_settlement_rows(_trades())
        summary = summarize_segments(rows, ["probability_bucket"], min_rows=1)
        bucket = summary[summary["probability_bucket"].eq("40-50%")].iloc[0]

        self.assertEqual(bucket["rows"], 2)
        self.assertAlmostEqual(bucket["avg_predicted_no_probability"], 0.475)
        self.assertAlmostEqual(bucket["win_rate"], 0.5)
        self.assertAlmostEqual(bucket["calibration_error"], 0.025)

    def test_clv_bucket_math(self) -> None:
        rows = prepare_no_settlement_rows(_trades())
        clv = clv_vs_profit(rows)
        positive = clv[clv["clv_bucket"].eq("2-10c")].iloc[0]

        self.assertGreater(positive["positive_clv_rate"], 0.0)
        self.assertIn("profit", positive.index)

    def test_segment_summaries_rank_failures(self) -> None:
        rows = prepare_no_settlement_rows(_trades())
        failures = failure_segments(rows, min_rows=1)

        self.assertFalse(failures.empty)
        self.assertIn("failure_score", failures.columns)
        self.assertIn("recommended_fix", failures.columns)

    def test_prior_period_only_suppression(self) -> None:
        rows = prepare_no_settlement_rows(_trades())
        rule = SuppressionRule(
            group_columns=tuple(),
            min_prior_rows=2,
            min_prior_profit_per_share=0.0,
            min_prior_avg_clv_cents=-10.0,
            min_prior_positive_clv_rate=0.0,
        )
        selected = apply_prior_suppression_rule(rows, rule)

        self.assertFalse(bool(selected.loc[0, "selected_by_rule"]))
        self.assertFalse(bool(selected.loc[1, "selected_by_rule"]))
        self.assertEqual(selected.loc[1, "prior_rows"], 1)
        self.assertEqual(selected.loc[2, "prior_rows"], 2)

    def test_no_future_leakage_in_group_suppression(self) -> None:
        rows = prepare_no_settlement_rows(_trades())
        rule = SuppressionRule(
            group_columns=("yes_team_abbr",),
            min_prior_rows=2,
            min_prior_profit_per_share=-1.0,
            min_prior_avg_clv_cents=-10.0,
            min_prior_positive_clv_rate=0.0,
        )
        selected = apply_prior_suppression_rule(rows, rule)
        first_aaa = selected[selected["yes_team_abbr"].eq("AAA")].iloc[0]
        third_aaa = selected[selected["yes_team_abbr"].eq("AAA")].iloc[2]

        self.assertFalse(bool(first_aaa["selected_by_rule"]))
        self.assertTrue(bool(third_aaa["selected_by_rule"]))

    def test_fair_price_and_parlay_remain_blocked(self) -> None:
        summary, *_ = build_no_settlement_calibration_audit(
            _trades(),
            proof_summary={"status": "not_proven", "single_game_edge_proven": False},
            fair_price_summary={"bets": 0},
            parlay_summary={"status": "blocked_single_game_edge_not_proven", "parlay_recommendations_allowed": False},
        )

        self.assertFalse(summary["single_game_edge_proven"])
        self.assertEqual(summary["fair_price_bets"], 0)
        self.assertEqual(summary["parlay_status"], "blocked_single_game_edge_not_proven")
        self.assertFalse(summary["parlay_recommendations_allowed"])


if __name__ == "__main__":
    unittest.main()
