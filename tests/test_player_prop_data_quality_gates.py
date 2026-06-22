from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.player_prop_data_quality_gates import (  # noqa: E402
    THRESHOLDS,
    build_gates_summary,
    compute_quality_metrics,
    evaluate_gates,
    write_data_quality_gate_reports,
)


PROP_TYPES = ["points", "rebounds", "assists"]
BOOKMAKERS = ["draftkings", "fanduel", "betmgm"]


def _nba_frame(n: int = 600, *, settled: int = 0, closing: int = 0) -> pd.DataFrame:
    # Rows cycle through 50 players x 3 prop types x 3 bookmakers, always
    # quoting the main line (25.5) so main-line row counts equal row counts.
    rows = []
    for i in range(n):
        rows.append(
            {
                "snapshot_time": f"2026-06-10T{i % 24:02d}:00:00+00:00",
                "league": "NBA",
                "canonical_game_key": "basketball|NBA|2026-06-10|NYK|SAS",
                "player_name": f"Player {i % 50}",
                "prop_type": PROP_TYPES[i % len(PROP_TYPES)],
                "line": 25.5,
                "over_price": 1.91,
                "under_price": 1.91,
                "bookmaker": BOOKMAKERS[i % len(BOOKMAKERS)],
                "source": "odds_api",
                "market_id": f"m{i}",
                "player_matched": True,
                "game_matched": True,
                "is_closing_snapshot": i < closing,
                "settlement_status": "settled" if i < settled else "pending_result",
            }
        )
    return pd.DataFrame(rows)


def _line_quality(markets: int = 100, *, with_closing: int = 0) -> pd.DataFrame:
    # Mirror the _nba_frame market keys so row-level main-line annotation joins.
    rows = []
    for i in range(markets):
        rows.append(
            {
                "league": "NBA",
                "player_name": f"Player {i % 50}",
                "prop_type": PROP_TYPES[i % len(PROP_TYPES)],
                "bookmaker": BOOKMAKERS[i % len(BOOKMAKERS)],
                "canonical_game_key": "basketball|NBA|2026-06-10|NYK|SAS",
                "likely_main_line": 25.5,
                "line_quality_label": "clean",
                "has_closing_snapshot": i < with_closing,
            }
        )
    return pd.DataFrame(rows)


class MetricsTests(unittest.TestCase):
    def test_metrics_computed(self) -> None:
        metrics = compute_quality_metrics(_nba_frame(600, settled=10, closing=5), _line_quality(100, with_closing=20), {"nba_markets_with_clv": 3})
        self.assertEqual(metrics["nba_snapshots"], 600)
        self.assertEqual(metrics["player_match_rate"], 1.0)
        self.assertEqual(metrics["settled_props"], 10)
        self.assertEqual(metrics["closing_like_snapshots"], 5)
        self.assertEqual(metrics["main_line_rate"], 1.0)
        self.assertEqual(metrics["closing_market_rate"], 0.2)
        self.assertEqual(metrics["clv_markets"], 3)
        self.assertEqual(metrics["bookmakers"], 3)
        self.assertEqual(metrics["prop_types"], 3)
        self.assertEqual(metrics["missing_core_field_rate"], 0.0)
        self.assertIn("points", metrics["settlement_by_prop_type"])

    def test_empty_is_not_ready(self) -> None:
        evaluation = evaluate_gates(compute_quality_metrics(pd.DataFrame(), pd.DataFrame(), {}))
        self.assertEqual(evaluation["status"], "not_ready")
        self.assertTrue(evaluation["blockers"])


class LadderTests(unittest.TestCase):
    def test_collection_ready_without_settlement(self) -> None:
        summary = build_gates_summary(_nba_frame(600), _line_quality(100), {})
        self.assertEqual(summary["status"], "collection_ready")
        self.assertTrue(any("settled_props" in b for b in summary["blockers"]))
        self.assertFalse(summary["approved"])

    def test_settlement_ready_needs_a_settled_prop(self) -> None:
        summary = build_gates_summary(_nba_frame(600, settled=5), _line_quality(100), {})
        self.assertEqual(summary["status"], "settlement_ready")

    def test_clv_ready_needs_closing_and_clv_pairs(self) -> None:
        summary = build_gates_summary(
            _nba_frame(600, settled=5, closing=60),
            _line_quality(100, with_closing=15),
            {"nba_markets_with_clv": 10},
        )
        self.assertEqual(summary["status"], "clv_ready")

    def test_modeling_ready_requires_real_samples(self) -> None:
        # 150 markets fully cover the 50x3x3 fixture market keys, so every
        # settled/closing row counts as a main-line row.
        summary = build_gates_summary(
            _nba_frame(600, settled=600, closing=200),
            _line_quality(150, with_closing=60),
            {"nba_markets_with_clv": THRESHOLDS["min_clv_markets_for_modeling"]},
        )
        self.assertEqual(summary["status"], "modeling_experiment_ready")

    def test_modeling_blocked_when_main_line_rows_thin(self) -> None:
        # Plenty settled overall, but only 100 of 150 markets join a main line,
        # so settled main-line rows fall below the 500-row v2 threshold.
        summary = build_gates_summary(
            _nba_frame(600, settled=600, closing=200),
            _line_quality(100, with_closing=40),
            {"nba_markets_with_clv": THRESHOLDS["min_clv_markets_for_modeling"]},
        )
        self.assertNotEqual(summary["status"], "modeling_experiment_ready")
        self.assertTrue(any("settled_main_line_rows" in b for b in summary["blockers"]))

    def test_modeling_not_granted_on_small_samples(self) -> None:
        summary = build_gates_summary(
            _nba_frame(600, settled=20, closing=200),
            _line_quality(100, with_closing=30),
            {"nba_markets_with_clv": 5},
        )
        self.assertEqual(summary["status"], "clv_ready")
        self.assertTrue(any("settled_props_for_modeling" in b for b in summary["blockers"]))

    def test_unstarted_games_pass_game_match_provisionally(self) -> None:
        frame = _nba_frame(600)
        frame["game_start_time"] = "2030-01-01T00:00:00+00:00"
        frame["game_matched"] = False  # cannot match an unplayed game
        summary = build_gates_summary(frame, _line_quality(100), {})
        self.assertIsNone(summary["metrics"]["game_match_rate"])
        self.assertEqual(summary["status"], "collection_ready")
        checks = {c["check"]: c for c in summary["checks"]["collection"]}
        self.assertTrue(checks["game_match_rate"]["passed"])
        self.assertIn("WARNING", checks["game_match_rate"]["detail"])

    def test_started_games_with_bad_game_match_block_collection(self) -> None:
        frame = _nba_frame(600)
        frame["game_start_time"] = "2020-01-01T00:00:00+00:00"
        frame["game_matched"] = False
        summary = build_gates_summary(frame, _line_quality(100), {})
        self.assertEqual(summary["metrics"]["game_match_rate"], 0.0)
        self.assertEqual(summary["status"], "not_ready")

    def test_low_player_match_blocks_collection(self) -> None:
        frame = _nba_frame(600)
        frame.loc[: int(len(frame) * 0.2), "player_matched"] = False
        summary = build_gates_summary(frame, _line_quality(100), {})
        self.assertEqual(summary["status"], "not_ready")


class WriteReportsTests(unittest.TestCase):
    def test_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            _nba_frame(600).to_csv(processed / "player_prop_snapshots_enriched.csv", index=False)

            summary = write_data_quality_gate_reports(root)

            reports = root / "data" / "reports"
            self.assertTrue((reports / "player_prop_data_quality_gates.json").exists())
            self.assertTrue((reports / "player_prop_data_quality_gates.md").exists())
            loaded = json.loads(
                (reports / "player_prop_data_quality_gates.json").read_text(encoding="utf-8")
            )
            self.assertIn(loaded["status"], loaded["status_ladder"])
            md = (reports / "player_prop_data_quality_gates.md").read_text(encoding="utf-8")
            self.assertIn("Research-only", md)
            self.assertFalse(summary["approved"])


if __name__ == "__main__":
    unittest.main()
