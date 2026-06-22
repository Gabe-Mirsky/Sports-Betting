from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.player_prop_schema import (  # noqa: E402
    PLAYER_PROP_SNAPSHOT_COLUMNS,
    build_player_prop_schema_outputs,
    empty_player_prop_snapshot_frame,
    validate_player_prop_snapshots,
    write_player_prop_template,
)


def _valid_row() -> dict:
    return {
        "snapshot_time": "2026-01-15 22:00:00",
        "sport": "basketball",
        "league": "NBA",
        "season": "2025-26",
        "game_date": "2026-01-15",
        "game_start_time": "2026-01-16 00:30:00",
        "canonical_game_key": "basketball|NBA|2026-01-15|OKC|HOU",
        "player_name": "Shai Gilgeous-Alexander",
        "player_id": "1628983",
        "team": "OKC",
        "opponent": "HOU",
        "home_away": "home",
        "prop_type": "points",
        "line": 31.5,
        "over_price": 1.91,
        "under_price": 1.91,
        "bookmaker": "kalshi",
        "source": "kalshi",
        "market_id": "KXNBAPTS-26JAN15-SGA-31",
        "is_closing_snapshot": True,
        "minutes_to_game_start": 30.0,
        "has_result": True,
        "actual_stat_value": 35.0,
        "over_won": True,
        "under_won": False,
        "raw_source_file": "data/raw/prop_odds/NBA/kalshi/20260115T220000Z__demo.json",
    }


class TemplateTests(unittest.TestCase):
    def test_template_has_exact_schema_columns_and_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = write_player_prop_template(Path(folder) / "template.csv")
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), list(PLAYER_PROP_SNAPSHOT_COLUMNS))
            self.assertEqual(len(frame), 0)

    def test_build_outputs_writes_template_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            templates = Path(folder) / "templates"
            reports = Path(folder) / "reports"
            summary = build_player_prop_schema_outputs(templates, reports)
            self.assertTrue((templates / "player_prop_snapshot_template.csv").exists())
            report_text = (reports / "player_prop_schema_summary.md").read_text(encoding="utf-8")
            for column in PLAYER_PROP_SNAPSHOT_COLUMNS:
                self.assertIn(column, report_text)
            self.assertTrue(summary["research_only"])
            self.assertFalse(summary["approved"])


class ValidationTests(unittest.TestCase):
    def test_valid_frame_passes(self) -> None:
        frame = pd.DataFrame([_valid_row()])
        result = validate_player_prop_snapshots(frame)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["errors"], [])

    def test_empty_frame_with_schema_columns_is_valid(self) -> None:
        result = validate_player_prop_snapshots(empty_player_prop_snapshot_frame())
        self.assertTrue(result["valid"])
        self.assertEqual(result["rows"], 0)

    def test_missing_column_is_an_error(self) -> None:
        frame = pd.DataFrame([_valid_row()]).drop(columns=["line"])
        result = validate_player_prop_snapshots(frame)
        self.assertFalse(result["valid"])
        self.assertTrue(any("missing columns" in e for e in result["errors"]))

    def test_invalid_home_away_is_an_error(self) -> None:
        row = _valid_row()
        row["home_away"] = "neutral"
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertFalse(result["valid"])
        self.assertTrue(any("home_away" in e for e in result["errors"]))

    def test_unknown_prop_type_is_a_warning_not_error(self) -> None:
        row = _valid_row()
        row["prop_type"] = "dunks"
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertTrue(result["valid"])
        self.assertTrue(any("prop_type" in w for w in result["warnings"]))

    def test_non_numeric_line_is_an_error(self) -> None:
        row = _valid_row()
        row["line"] = "thirty"
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertFalse(result["valid"])
        self.assertTrue(any("line" in e for e in result["errors"]))

    def test_has_result_without_actual_is_an_error(self) -> None:
        row = _valid_row()
        row["actual_stat_value"] = None
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertFalse(result["valid"])
        self.assertTrue(any("actual_stat_value" in e for e in result["errors"]))

    def test_settlement_inconsistency_is_an_error(self) -> None:
        row = _valid_row()
        row["actual_stat_value"] = 20.0  # below the 31.5 line but over_won=True
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertFalse(result["valid"])
        self.assertTrue(any("over_won" in e for e in result["errors"]))

    def test_malformed_canonical_key_is_an_error(self) -> None:
        row = _valid_row()
        row["canonical_game_key"] = "OKC vs HOU"
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertFalse(result["valid"])
        self.assertTrue(any("canonical_game_key" in e for e in result["errors"]))

    def test_missing_required_value_is_an_error(self) -> None:
        row = _valid_row()
        row["bookmaker"] = ""
        result = validate_player_prop_snapshots(pd.DataFrame([row]))
        self.assertFalse(result["valid"])
        self.assertTrue(any("bookmaker" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
