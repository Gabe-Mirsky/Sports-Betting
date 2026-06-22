from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.historical_inventory_quality import build_quality_audit, write_quality_audit  # noqa: E402

TODAY = date(2026, 6, 14)
INV_COLUMNS = [
    "canonical_game_key", "league", "game_date", "home_team", "away_team", "source",
    "outcome_available", "game_odds_available", "prop_odds_available", "closing_available",
]


def _inv_row(key, league, gdate, home, away, source="football_data",
             outcome=True, game_odds=False, prop=False, closing=False):
    return {
        "canonical_game_key": key, "league": league, "game_date": gdate,
        "home_team": home, "away_team": away, "source": source,
        "outcome_available": outcome, "game_odds_available": game_odds,
        "prop_odds_available": prop, "closing_available": closing,
    }


def _write(tmp: Path, inv_rows, staging_rows=None):
    inv_path = tmp / "historical_game_inventory.csv"
    pd.DataFrame(inv_rows, columns=INV_COLUMNS).to_csv(inv_path, index=False)
    staging_dir = tmp / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    if staging_rows is not None:
        pd.DataFrame(staging_rows).to_csv(staging_dir / "retrosheet_MLB.csv", index=False)
    return inv_path, staging_dir


class TestQualityAudit(unittest.TestCase):
    def test_clean_inventory_is_clean(self) -> None:
        rows = [
            _inv_row("basketball|NBA|2024-01-01|LAL|BOS", "NBA", "2024-01-01", "LAL", "BOS",
                     outcome=True, game_odds=True, closing=True),
            _inv_row("baseball|MLB|2024-05-02|NYA|BOS", "MLB", "2024-05-02", "NYA", "BOS",
                     source="retrosheet", outcome=True),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            inv_path, staging_dir = _write(Path(tmp), rows)
            audit = build_quality_audit(inv_path, staging_dir, today=TODAY)
        self.assertEqual(audit["verdict"], "clean")
        self.assertEqual(audit["checks"]["counts_by_status"]["closing_recorded"], 1)
        self.assertEqual(audit["checks"]["counts_by_status"]["outcome_only"], 1)
        self.assertEqual(audit["checks"]["counts_by_status"]["settled"], 0)

    def test_warnings_downgrade_verdict(self) -> None:
        rows = [
            # closing flagged but no game odds -> warning
            _inv_row("soccer|EPL|2010-08-14|ARSENAL|CHELSEA", "EPL", "2010-08-14", "Arsenal", "Chelsea",
                     outcome=True, game_odds=False, closing=True),
            # future date -> warning
            _inv_row("soccer|EPL|2030-08-14|SPURS|EVERTON", "EPL", "2030-08-14", "Spurs", "Everton",
                     outcome=False, game_odds=True),
            # WORLD_CUP label needs normalization (key is WORLDCUP) -> warning
            _inv_row("soccer|WORLDCUP|2022-12-18|ARGENTINA|FRANCE", "WORLD_CUP", "2022-12-18",
                     "Argentina", "France", source="world_cup_results", outcome=True),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            inv_path, staging_dir = _write(Path(tmp), rows)
            audit = build_quality_audit(inv_path, staging_dir, today=TODAY)
        self.assertEqual(audit["verdict"], "usable_with_warnings")
        self.assertEqual(audit["checks"]["closing_without_game_odds"]["count"], 1)
        self.assertEqual(audit["checks"]["dates"]["future"], 1)
        self.assertIn("WORLD_CUP", audit["checks"]["league_normalization"]["labels_needing_normalization"])
        self.assertEqual(audit["checks"]["league_normalization"]["key_vs_label_mismatch"], 0)
        self.assertFalse(audit["critical_issues"])

    def test_prop_closing_is_documented_exception_without_game_odds(self) -> None:
        rows = [
            _inv_row(
                "basketball|NBA|2025-01-01|LAL|BOS",
                "NBA",
                "2025-01-01",
                "LAL",
                "BOS",
                source="sportsgameodds",
                outcome=True,
                game_odds=False,
                prop=True,
                closing=True,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            inv_path, staging_dir = _write(Path(tmp), rows)
            audit = build_quality_audit(inv_path, staging_dir, today=TODAY)
        closing_check = audit["checks"]["closing_without_game_odds"]
        self.assertEqual(audit["verdict"], "clean")
        self.assertEqual(closing_check["count"], 0)
        self.assertEqual(closing_check["total_including_documented_exceptions"], 1)
        self.assertEqual(closing_check["documented_prop_closing_exceptions"], 1)

    def test_duplicate_key_is_critical(self) -> None:
        key = "baseball|MLB|2024-05-02|NYA|BOS"
        rows = [
            _inv_row(key, "MLB", "2024-05-02", "NYA", "BOS", source="retrosheet"),
            _inv_row(key, "MLB", "2024-05-02", "NYA", "BOS", source="retrosheet"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            inv_path, staging_dir = _write(Path(tmp), rows)
            audit = build_quality_audit(inv_path, staging_dir, today=TODAY)
        self.assertEqual(audit["checks"]["duplicate_canonical_keys"]["excess_rows"], 1)
        self.assertEqual(audit["verdict"], "needs_fix")

    def test_missing_field_is_critical(self) -> None:
        rows = [_inv_row("baseball|MLB|2024-05-02|NYA|BOS", "MLB", "2024-05-02", "", "BOS")]
        with tempfile.TemporaryDirectory() as tmp:
            inv_path, staging_dir = _write(Path(tmp), rows)
            audit = build_quality_audit(inv_path, staging_dir, today=TODAY)
        self.assertEqual(audit["checks"]["missing_fields"]["home_team"], 1)
        self.assertEqual(audit["verdict"], "needs_fix")

    def test_mlb_doubleheader_collisions_from_staging(self) -> None:
        rows = [_inv_row("baseball|MLB|2024-05-02|NYA|BOS", "MLB", "2024-05-02", "NYA", "BOS",
                         source="retrosheet")]
        # Two staging rows collapse to one canonical key (a doubleheader).
        staging = [
            {"canonical_game_key": "baseball|MLB|2024-05-02|NYA|BOS", "league": "MLB",
             "game_date": "2024-05-02", "home_team": "NYA", "away_team": "BOS"},
            {"canonical_game_key": "baseball|MLB|2024-05-02|NYA|BOS", "league": "MLB",
             "game_date": "2024-05-02", "home_team": "NYA", "away_team": "BOS"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            inv_path, staging_dir = _write(Path(tmp), rows, staging_rows=staging)
            audit = build_quality_audit(inv_path, staging_dir, today=TODAY)
        self.assertEqual(audit["checks"]["mlb_doubleheader_collisions"]["count"], 1)

    def test_write_quality_audit_emits_json_and_md(self) -> None:
        rows = [_inv_row("basketball|NBA|2024-01-01|LAL|BOS", "NBA", "2024-01-01", "LAL", "BOS",
                         outcome=True, game_odds=True)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inv_path, staging_dir = _write(tmp_path, rows)
            reports_dir = tmp_path / "reports"
            paths = write_quality_audit(reports_dir, inventory_path=inv_path, staging_dir=staging_dir)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            data = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(data["report"], "historical_game_inventory_quality_audit")
            self.assertIn("Quality Audit", paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
