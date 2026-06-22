from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.player_prop_settlement_outcomes import (  # noqa: E402
    SMALL_SAMPLE_THRESHOLD,
    build_settlement_outcomes,
    write_settlement_outcome_reports,
)


GAME_KEY = "basketball|NBA|2026-06-08|OKC|HOU"


def _settled_row(**overrides) -> dict:
    row = {
        "snapshot_time": "2026-06-08T18:00:00+00:00",
        "league": "NBA",
        "canonical_game_key": GAME_KEY,
        "game_date": "2026-06-08",
        "game_start_time": "2026-06-09T00:30:00+00:00",
        "player_name": "Test Player",
        "team": "OKC",
        "prop_type": "points",
        "line": 25.5,
        "actual_stat_value": 30.0,
        "over_won": True,
        "under_won": False,
        "push": False,
        "bookmaker": "draftkings",
        "source": "odds_api",
        "is_closing_snapshot": False,
        "minutes_to_game_start": 390.0,
        "settlement_status": "settled",
    }
    row.update(overrides)
    return row


def _pending_row(**overrides) -> dict:
    row = _settled_row(
        actual_stat_value=pd.NA,
        over_won=pd.NA,
        under_won=pd.NA,
        push=pd.NA,
        settlement_status="pending_result",
        game_start_time="2030-01-01T00:30:00+00:00",
    )
    row.update(overrides)
    return row


class BuildOutcomesTests(unittest.TestCase):
    def test_pending_only_reports_reason_without_forcing(self) -> None:
        enriched = pd.DataFrame([_pending_row(), _pending_row(player_name="Other Player")])
        result = build_settlement_outcomes(enriched, pd.DataFrame())
        summary = result["summary"]
        self.assertEqual(summary["settled_props"], 0)
        self.assertEqual(summary["pending_props"], 2)
        self.assertEqual(len(summary["pending_games"]), 1)
        self.assertIn("not been played", summary["pending_games"][0]["reason"])
        self.assertTrue(result["outcomes"].empty)
        self.assertTrue(summary["research_only"])
        self.assertFalse(summary["approved"])

    def test_started_pending_game_points_to_download_refresh(self) -> None:
        enriched = pd.DataFrame([_pending_row(game_start_time="2020-01-01T00:30:00+00:00")])
        summary = build_settlement_outcomes(enriched, pd.DataFrame())["summary"]
        self.assertIn("--download", summary["pending_games"][0]["reason"])

    def test_settled_splits_and_small_sample_warning(self) -> None:
        enriched = pd.DataFrame(
            [
                _settled_row(),
                _settled_row(
                    player_name="Under Guy",
                    actual_stat_value=20.0,
                    over_won=False,
                    under_won=True,
                    bookmaker="fanduel",
                    is_closing_snapshot=True,
                    prop_type="rebounds",
                    line=22.5,
                ),
                _settled_row(
                    player_name="Push Guy",
                    line=30.0,
                    actual_stat_value=30.0,
                    over_won=False,
                    under_won=False,
                    push=True,
                ),
                _pending_row(player_name="Pending Guy"),
            ]
        )
        likely_main = pd.DataFrame(
            [
                {
                    "league": "NBA",
                    "player_name": "Test Player",
                    "prop_type": "points",
                    "bookmaker": "draftkings",
                    "canonical_game_key": GAME_KEY,
                    "likely_main_line": 25.5,
                }
            ]
        )
        result = build_settlement_outcomes(enriched, likely_main)
        summary = result["summary"]
        self.assertEqual(summary["settled_props"], 3)
        self.assertEqual(summary["pending_props"], 1)
        self.assertEqual(summary["overall"]["over_won"], 1)
        self.assertEqual(summary["overall"]["under_won"], 1)
        self.assertEqual(summary["overall"]["push"], 1)
        self.assertTrue(summary["small_sample"])
        self.assertTrue(any("SMALL SAMPLE" in w for w in summary["warnings"]))
        self.assertLess(summary["settled_props"], SMALL_SAMPLE_THRESHOLD)

        by_book = {record["bookmaker"]: record for record in summary["by_bookmaker"]}
        self.assertEqual(by_book["fanduel"]["under_won"], 1)
        main_alt = summary["main_line_vs_alt"]
        self.assertEqual(main_alt["likely_main_line"]["settled"], 1)
        self.assertEqual(main_alt["alternate_or_unresolved"]["settled"], 2)
        closing = summary["closing_vs_early"]
        self.assertEqual(closing["closing_like"]["settled"], 1)
        self.assertEqual(closing["early"]["settled"], 2)

        outcomes = result["outcomes"]
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(set(outcomes["outcome"]), {"over_won", "under_won", "push"})

    def test_empty_input(self) -> None:
        result = build_settlement_outcomes(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(result["summary"]["settled_props"], 0)
        self.assertTrue(result["summary"]["warnings"])


class WriteReportsTests(unittest.TestCase):
    def test_writes_all_outputs_even_with_nothing_settled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame([_pending_row()]).to_csv(
                processed / "player_prop_snapshots_enriched.csv", index=False
            )

            summary = write_settlement_outcome_reports(root)

            reports = root / "data" / "reports"
            for filename in (
                "player_prop_settlement_outcomes_summary.json",
                "player_prop_settlement_outcomes.csv",
                "player_prop_settlement_outcomes.md",
            ):
                self.assertTrue((reports / filename).exists(), filename)
            loaded = json.loads(
                (reports / "player_prop_settlement_outcomes_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["settled_props"], 0)
            self.assertEqual(loaded["pending_props"], 1)
            md = (reports / "player_prop_settlement_outcomes.md").read_text(encoding="utf-8")
            self.assertIn("No Settled Props Yet", md)
            self.assertIn("Research-only", md)
            self.assertFalse(summary["approved"])


if __name__ == "__main__":
    unittest.main()
