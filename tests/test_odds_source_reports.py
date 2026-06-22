"""Tests for the multi-source reports, dashboard section, and pipeline plan."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_cross_source_prop_comparison as cross  # noqa: E402
import run_full_prop_pipeline as pipeline  # noqa: E402
from reports.dashboard import _build_odds_sources_section  # noqa: E402


def _market_row(source: str, book: str, line: float, over: float, under: float,
                snapshot: str = "2026-06-11T17:00:00+00:00", player: str = "jose alvarado",
                game: str = "basketball|NBA|2026-06-13|SAS|NYK") -> dict:
    return {
        "source": source,
        "canonical_game_key": game,
        "player_norm": player,
        "prop_type": "points",
        "book_norm": book,
        "line": line,
        "over_price": over,
        "under_price": under,
        "snapshot_time": snapshot,
        "snapshot_ts": pd.Timestamp(snapshot),
    }


class BookmakerNormalizationTests(unittest.TestCase):
    def test_caesars_alias(self) -> None:
        self.assertEqual(cross.normalize_bookmaker("williamhill_us"), "caesars")
        self.assertEqual(cross.normalize_bookmaker("FanDuel"), "fanduel")

    def test_player_normalization(self) -> None:
        self.assertEqual(cross.normalize_player("De'Aaron Fox"), "deaaron fox")
        self.assertEqual(cross.normalize_player("  Jose  Alvarado "), "jose alvarado")


class CrossSourceComparisonTests(unittest.TestCase):
    def test_overlap_detected_and_diffs_computed(self) -> None:
        latest = pd.DataFrame([
            _market_row("odds_api", "fanduel", 4.5, 2.00, 1.80),
            _market_row("sportsgameodds", "fanduel", 4.5, 2.02, 1.74,
                        snapshot="2026-06-11T17:30:00+00:00"),
        ])
        comparison, summary = cross.build_comparison(latest)
        self.assertTrue(summary["overlap_found"])
        self.assertEqual(len(comparison), 1)
        pair = summary["pairs"][0]
        self.assertEqual(pair["shared_games"], 1)
        self.assertEqual(pair["exact_market_overlap_rows"], 1)
        self.assertAlmostEqual(pair["mean_abs_over_price_diff"], 0.02, places=6)
        self.assertEqual(pair["fresher_counts"]["sportsgameodds"], 1)

    def test_line_disagreement_counted(self) -> None:
        latest = pd.DataFrame([
            _market_row("odds_api", "betmgm", 4.5, 2.0, 1.8),
            _market_row("sportsgameodds", "betmgm", 3.5, 1.77, 1.95),
        ])
        _, summary = cross.build_comparison(latest)
        pair = summary["pairs"][0]
        self.assertEqual(pair["exact_market_overlap_rows"], 0)
        self.assertEqual(pair["line_disagreement_markets"], 1)
        self.assertTrue(summary["line_disagreements_sample"])

    def test_no_overlap_still_reports_reason(self) -> None:
        latest = pd.DataFrame([
            _market_row("odds_api", "fanduel", 4.5, 2.0, 1.8,
                        game="basketball|NBA|2026-06-10|NYK|SAS"),
            _market_row("sportsgameodds", "fanduel", 4.5, 2.0, 1.8),
        ])
        comparison, summary = cross.build_comparison(latest)
        self.assertFalse(summary["overlap_found"])
        self.assertIn("No shared canonical games", summary["reason"])
        self.assertTrue(comparison.empty)
        self.assertIn("coverage_by_source", summary)

    def test_single_source_explained(self) -> None:
        latest = pd.DataFrame([_market_row("odds_api", "fanduel", 4.5, 2.0, 1.8)])
        _, summary = cross.build_comparison(latest)
        self.assertFalse(summary["overlap_found"])
        self.assertIn("only one source", summary["reason"])

    def test_empty_frame(self) -> None:
        _, summary = cross.build_comparison(pd.DataFrame())
        self.assertFalse(summary["overlap_found"])


class DashboardSectionTests(unittest.TestCase):
    def test_renders_without_any_reports(self) -> None:
        empty_dir = PROJECT_ROOT / "data" / "reports" / "_test_empty_sources_dir"
        empty_dir.mkdir(parents=True, exist_ok=True)
        try:
            html = _build_odds_sources_section(empty_dir)
            self.assertIn("Odds Sources (multi-source status)", html)
            self.assertIn("SportsGameOdds", html)
            self.assertIn("API-Sports (probe-only)", html)
        finally:
            import shutil

            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_renders_with_real_reports_when_present(self) -> None:
        reports = PROJECT_ROOT / "data" / "reports"
        if not (reports / "sportsgameodds_probe_summary.json").exists():
            self.skipTest("no real probe summary on disk")
        html = _build_odds_sources_section(reports)
        self.assertIn("Key works", html)
        self.assertIn("Cross-source comparison", html)
        self.assertIn("odds_source_comparison.md", html)


class PipelinePlanTests(unittest.TestCase):
    def test_new_steps_present_and_optional(self) -> None:
        steps = {step[0]: step for step in pipeline.PIPELINE_STEPS}
        for name in ("probe_sportsgameodds", "collect_sportsgameodds", "probe_apisports",
                     "odds_source_comparison", "odds_source_usage", "cross_source_comparison"):
            self.assertIn(name, steps)
            self.assertTrue(steps[name][4], f"{name} must be optional so a missing/failed "
                                            "script never breaks the pipeline")

    def test_probe_failure_cannot_break_pipeline(self) -> None:
        # Probes and source reports are kind "report"/"collection": run_full_prop_pipeline
        # continues past their failures by design. Verify the kinds are right.
        kinds = {step[0]: step[3] for step in pipeline.PIPELINE_STEPS}
        self.assertEqual(kinds["probe_sportsgameodds"], "report")
        self.assertEqual(kinds["probe_apisports"], "report")
        self.assertEqual(kinds["collect_sportsgameodds"], "collection")

    def test_sgo_probe_uses_cheap_mode(self) -> None:
        args = {step[0]: step[2] for step in pipeline.PIPELINE_STEPS}
        self.assertIn("--cheap", args["probe_sportsgameodds"])
        self.assertIn("--max-age-hours", args["probe_apisports"])

    def test_order_collection_before_reports(self) -> None:
        names = [step[0] for step in pipeline.PIPELINE_STEPS]
        self.assertLess(names.index("collect_sportsgameodds"), names.index("enrich_snapshots"))
        self.assertLess(names.index("odds_source_usage"), names.index("dashboard"))


if __name__ == "__main__":
    unittest.main()
