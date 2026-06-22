from __future__ import annotations

import unittest

import pandas as pd

from src.strategy.single_game_edge_diagnostics import (
    build_single_game_edge_diagnostics,
    build_walk_forward_slices,
    prepare_diagnostics_rows,
    summarize_slice,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "game_id": "1",
                "market_ticker": "A",
                "yes_team_abbr": "AAA",
                "trade": True,
                "side": "YES",
                "candidate_side": "YES",
                "actual_yes_win": True,
                "profit": 0.40,
                "cost": 0.60,
                "shares": 1,
                "price_cents": 60,
                "clv_cents": 2,
                "edge": 0.07,
                "model_prob": 0.70,
                "market_prob": 0.60,
                "volume": 200,
                "snapshot_target": "pregame_60m",
            },
            {
                "date": "2026-01-02",
                "game_id": "2",
                "market_ticker": "B",
                "yes_team_abbr": "BBB",
                "trade": True,
                "side": "YES",
                "candidate_side": "YES",
                "actual_yes_win": False,
                "profit": -0.55,
                "cost": 0.55,
                "shares": 1,
                "price_cents": 55,
                "clv_cents": -3,
                "edge": 0.08,
                "model_prob": 0.63,
                "market_prob": 0.55,
                "volume": 50,
                "snapshot_target": "pregame_60m",
            },
            {
                "date": "2026-01-03",
                "game_id": "3",
                "market_ticker": "C",
                "yes_team_abbr": "CCC",
                "trade": True,
                "side": "NO",
                "candidate_side": "NO",
                "actual_yes_win": False,
                "profit": 0.70,
                "cost": 0.30,
                "shares": 1,
                "price_cents": 30,
                "clv_cents": 1,
                "edge": 0.06,
                "model_prob": 0.75,
                "market_prob": 0.30,
                "volume": 500,
                "snapshot_target": "pregame_30m",
            },
        ]
    )


class TestSingleGameEdgeDiagnostics(unittest.TestCase):
    def test_bucket_summary_math(self) -> None:
        diagnostics = prepare_diagnostics_rows(_rows())
        overall = summarize_slice(diagnostics)
        row = overall.iloc[0]

        self.assertEqual(row["trade_count"], 3)
        self.assertAlmostEqual(row["profit"], 0.55)
        self.assertAlmostEqual(row["amount_risked"], 1.45)
        self.assertAlmostEqual(row["roi_on_amount_risked"], 0.55 / 1.45)
        self.assertAlmostEqual(row["average_profit_per_trade"], 0.55 / 3)
        self.assertAlmostEqual(row["average_clv_cents"], 0.0)
        self.assertAlmostEqual(row["positive_clv_rate"], 2 / 3)
        self.assertAlmostEqual(row["brier_score"], ((0.70 - 1) ** 2 + (0.63 - 0) ** 2 + (0.75 - 1) ** 2) / 3)

    def test_yes_no_split_uses_contract_side(self) -> None:
        diagnostics = prepare_diagnostics_rows(_rows())
        by_side = summarize_slice(diagnostics, ["diagnostic_side"], min_rows=1)

        yes = by_side[by_side["diagnostic_side"].eq("YES")].iloc[0]
        no = by_side[by_side["diagnostic_side"].eq("NO")].iloc[0]

        self.assertEqual(yes["trade_count"], 2)
        self.assertAlmostEqual(yes["profit"], -0.15)
        self.assertAlmostEqual(yes["average_clv_cents"], -0.5)
        self.assertEqual(no["trade_count"], 1)
        self.assertAlmostEqual(no["profit"], 0.70)
        self.assertAlmostEqual(no["calibration_error"], 0.25)

    def test_small_sample_warning(self) -> None:
        diagnostics = prepare_diagnostics_rows(_rows())
        by_side = summarize_slice(diagnostics, ["diagnostic_side"], min_rows=5)

        self.assertTrue(by_side["small_sample_warning"].all())

    def test_walk_forward_slice_logic_requires_prior_and_eval_success(self) -> None:
        rows = []
        for index, (date, clv) in enumerate(
            [
                ("2026-01-01", 2),
                ("2026-02-01", 1),
                ("2026-03-01", 3),
                ("2026-04-01", -4),
            ],
            start=1,
        ):
            rows.append(
                {
                    "date": date,
                    "game_id": str(index),
                    "market_ticker": f"T{index}",
                    "trade": True,
                    "side": "YES",
                    "candidate_side": "YES",
                    "actual_yes_win": True,
                    "profit": 0.20,
                    "cost": 0.80,
                    "shares": 1,
                    "price_cents": 80,
                    "clv_cents": clv,
                    "edge": 0.10,
                    "model_prob": 0.90,
                    "market_prob": 0.80,
                    "volume": 100,
                    "snapshot_target": "pregame_60m",
                }
            )
        diagnostics = prepare_diagnostics_rows(pd.DataFrame(rows))
        walk_forward = build_walk_forward_slices(
            diagnostics,
            segment_sets=[["diagnostic_side"]],
            min_prior_rows=2,
            min_eval_rows=1,
        )

        march = walk_forward[walk_forward["month"].eq("2026-03")].iloc[0]
        april = walk_forward[walk_forward["month"].eq("2026-04")].iloc[0]

        self.assertTrue(march["prior_period_selected"])
        self.assertTrue(march["survived_eval"])
        self.assertTrue(april["prior_period_selected"])
        self.assertFalse(april["survived_eval"])
        self.assertEqual(april["status"], "research_only")

    def test_not_proven_summary_keeps_fair_price_and_parlay_blocked(self) -> None:
        fair = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "market_ticker": "A",
                    "recommendation": "No bet",
                    "ungated_recommendation": "Bet YES",
                    "confidence": "medium",
                    "spread": 1,
                    "proof_gate_status": "not_proven",
                }
            ]
        )
        summary, _, _, _, _ = build_single_game_edge_diagnostics(
            trades=_rows().iloc[[0]].copy(),
            fair_price_signals=fair,
            proof_summary={"status": "not_proven", "single_game_edge_proven": False, "failed_gates": ["average_clv"]},
            parlay_summary={"status": "blocked_single_game_edge_not_proven", "parlay_recommendations_allowed": False},
            min_rows=1,
        )

        self.assertFalse(summary["single_game_edge_proven"])
        self.assertEqual(summary["actionable_fair_price_bets"], 0)
        self.assertEqual(summary["ungated_fair_price_bets"], 1)
        self.assertFalse(summary["parlay_recommendations_allowed"])


if __name__ == "__main__":
    unittest.main()
