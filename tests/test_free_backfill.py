from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.free_backfill import (  # noqa: E402
    INVENTORY_COLUMNS,
    STAGING_COLUMNS,
    build_historical_game_inventory,
    coerce_staging_frame,
    run_importer_from_input,
    write_staging_frame,
)
from reports.free_backfill_audit import source_inventory_frame, write_source_audit  # noqa: E402
import backfill_free_soccer_results_and_odds as soccer  # noqa: E402


class TestStagingNormalization(unittest.TestCase):
    def test_coerce_infers_flags_and_builds_canonical_key(self) -> None:
        partial = pd.DataFrame(
            {
                "game_date": ["2026-06-13"],
                "home_team": ["Arsenal"],
                "away_team": ["Chelsea"],
                "home_score": [2],
                "away_score": [1],
                "home_odds": [1.8],
                "draw_odds": [3.5],
                "away_odds": [4.2],
            }
        )
        staged = coerce_staging_frame(partial, source="football_data", sport="soccer", league="EPL")
        self.assertEqual(list(staged.columns), STAGING_COLUMNS)
        row = staged.iloc[0]
        self.assertEqual(row["canonical_game_key"], "soccer|EPL|2026-06-13|ARSENAL|CHELSEA")
        self.assertTrue(bool(row["outcome_available"]))
        self.assertTrue(bool(row["game_odds_available"]))
        self.assertFalse(bool(row["prop_odds_available"]))
        self.assertFalse(bool(row["closing_available"]))

    def test_outcome_only_when_no_scores_or_odds(self) -> None:
        partial = pd.DataFrame(
            {"game_date": ["2026-06-13"], "home_team": ["Aaa"], "away_team": ["Bbb"]}
        )
        staged = coerce_staging_frame(partial, source="kaggle", sport="soccer", league="WORLD_CUP")
        row = staged.iloc[0]
        self.assertFalse(bool(row["outcome_available"]))
        self.assertFalse(bool(row["game_odds_available"]))

    def test_valid_odds_force_game_odds_available(self) -> None:
        partial = pd.DataFrame(
            {
                "game_date": ["2026-06-13"],
                "home_team": ["Arsenal"],
                "away_team": ["Chelsea"],
                "home_odds": [1.8],
                "game_odds_available": [False],
            }
        )
        staged = coerce_staging_frame(partial, source="football_data", sport="soccer", league="EPL")
        self.assertTrue(bool(staged.iloc[0]["game_odds_available"]))

    def test_stale_closing_flag_without_prices_is_demoted(self) -> None:
        partial = pd.DataFrame(
            {
                "game_date": ["2026-06-13"],
                "home_team": ["Arsenal"],
                "away_team": ["Chelsea"],
                "closing_available": [True],
            }
        )
        staged = coerce_staging_frame(partial, source="football_data", sport="soccer", league="EPL")
        row = staged.iloc[0]
        self.assertFalse(bool(row["game_odds_available"]))
        self.assertFalse(bool(row["closing_available"]))

    def test_prop_closing_is_documented_exception_without_game_odds(self) -> None:
        partial = pd.DataFrame(
            {
                "game_date": ["2026-06-13"],
                "home_team": ["LAL"],
                "away_team": ["BOS"],
                "prop_odds_available": [True],
                "closing_available": [True],
            }
        )
        staged = coerce_staging_frame(partial, source="sportsgameodds", sport="basketball", league="NBA")
        row = staged.iloc[0]
        self.assertFalse(bool(row["game_odds_available"]))
        self.assertTrue(bool(row["prop_odds_available"]))
        self.assertTrue(bool(row["closing_available"]))

    def test_invalid_rows_are_dropped(self) -> None:
        partial = pd.DataFrame(
            {
                "game_date": ["2026-06-13", "not-a-date"],
                "home_team": ["Arsenal", "Spurs"],
                "away_team": ["Chelsea", "Spurs"],  # 2nd row: home == away -> dropped
            }
        )
        staged = coerce_staging_frame(partial, source="x", sport="soccer", league="EPL")
        self.assertEqual(len(staged), 1)


class TestInventoryAggregation(unittest.TestCase):
    def test_build_inventory_ors_flags_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "staging"
            # Source A: outcome only.
            write_staging_frame(
                pd.DataFrame({"game_date": ["2026-06-13"], "home_team": ["Arsenal"],
                              "away_team": ["Chelsea"], "home_score": [2], "away_score": [1]}),
                source="kaggle", sport="soccer", league="EPL", out_dir=staging,
            )
            # Source B: same game, game odds only.
            write_staging_frame(
                pd.DataFrame({"game_date": ["2026-06-13"], "home_team": ["Arsenal"],
                              "away_team": ["Chelsea"], "home_odds": [1.8], "draw_odds": [3.5], "away_odds": [4.0]}),
                source="football_data", sport="soccer", league="EPL", out_dir=staging,
            )
            inventory = build_historical_game_inventory(staging, Path(tmp) / "inv.csv")

        self.assertEqual(list(inventory.columns), INVENTORY_COLUMNS)
        self.assertEqual(len(inventory), 1)
        row = inventory.iloc[0]
        self.assertTrue(bool(row["outcome_available"]))
        self.assertTrue(bool(row["game_odds_available"]))
        self.assertIn("kaggle", row["source"])
        self.assertIn("football_data", row["source"])

    def test_build_inventory_demotes_stale_closing_flags_from_existing_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "staging"
            staging.mkdir()
            pd.DataFrame(
                [
                    {
                        "source": "football_data",
                        "sport": "soccer",
                        "league": "EPL",
                        "game_date": "2026-06-13",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "home_score": 2,
                        "away_score": 1,
                        "home_odds": pd.NA,
                        "draw_odds": pd.NA,
                        "away_odds": pd.NA,
                        "outcome_available": True,
                        "game_odds_available": False,
                        "prop_odds_available": False,
                        "closing_available": True,
                        "canonical_game_key": "soccer|EPL|2026-06-13|ARSENAL|CHELSEA",
                    }
                ],
                columns=STAGING_COLUMNS,
            ).to_csv(staging / "football_data_EPL.csv", index=False)
            inventory = build_historical_game_inventory(staging, Path(tmp) / "inv.csv")

        row = inventory.iloc[0]
        self.assertFalse(bool(row["game_odds_available"]))
        self.assertFalse(bool(row["closing_available"]))

    def test_empty_staging_writes_header_only_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory = build_historical_game_inventory(Path(tmp) / "staging", Path(tmp) / "inv.csv")
        self.assertTrue(inventory.empty)
        self.assertEqual(list(inventory.columns), INVENTORY_COLUMNS)


class TestSoccerImporter(unittest.TestCase):
    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": ["13/06/2026"],  # dd/mm/yyyy
                "HomeTeam": ["Arsenal"],
                "AwayTeam": ["Chelsea"],
                "FTHG": [2],
                "FTAG": [1],
                "B365H": [1.8],
                "B365D": [3.5],
                "B365A": [4.2],
                "B365CH": [1.75],  # closing column present
                "B365CD": [3.4],
                "B365CA": [4.4],
            }
        )

    def test_normalize_maps_columns_and_closing(self) -> None:
        staged = coerce_staging_frame(
            soccer.normalize(self._raw()), source="football_data", sport="soccer", league="EPL"
        )
        row = staged.iloc[0]
        self.assertEqual(row["canonical_game_key"], "soccer|EPL|2026-06-13|ARSENAL|CHELSEA")
        self.assertTrue(bool(row["game_odds_available"]))
        self.assertTrue(bool(row["closing_available"]))

    def test_normalize_blank_closing_values_are_not_closing_available(self) -> None:
        raw = self._raw().astype({"B365CH": "object", "B365CD": "object", "B365CA": "object"})
        raw.loc[0, ["B365CH", "B365CD", "B365CA"]] = [pd.NA, "", None]
        staged = coerce_staging_frame(
            soccer.normalize(raw), source="football_data", sport="soccer", league="EPL"
        )
        row = staged.iloc[0]
        self.assertTrue(bool(row["game_odds_available"]))
        self.assertFalse(bool(row["closing_available"]))

    def test_normalize_valid_closing_values_are_closing_available(self) -> None:
        raw = self._raw()
        raw.loc[0, ["B365H", "B365D", "B365A"]] = [pd.NA, pd.NA, pd.NA]
        staged = coerce_staging_frame(
            soccer.normalize(raw), source="football_data", sport="soccer", league="EPL"
        )
        row = staged.iloc[0]
        self.assertTrue(bool(row["game_odds_available"]))
        self.assertTrue(bool(row["closing_available"]))
        self.assertEqual(float(row["home_odds"]), 1.75)

    def test_run_importer_from_input_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "E0.csv"
            self._raw().to_csv(raw_path, index=False)
            path, raw_rows = run_importer_from_input(
                input_path=raw_path, normalize_fn=soccer.normalize,
                source="football_data", sport="soccer", league="EPL", out_dir=Path(tmp) / "staging",
            )
            self.assertTrue(path.exists())
            self.assertEqual(raw_rows, 1)
            staged = pd.read_csv(path)
            self.assertEqual(staged.iloc[0]["canonical_game_key"], "soccer|EPL|2026-06-13|ARSENAL|CHELSEA")


class TestSourceAudit(unittest.TestCase):
    def test_inventory_has_required_columns(self) -> None:
        frame = source_inventory_frame()
        for column in [
            "league", "source_name", "source_type", "free_or_quota", "data_type",
            "date_range", "odds_available", "prop_odds_available", "closing_available",
            "license_notes", "importer_needed",
        ]:
            self.assertIn(column, frame.columns)
        self.assertGreaterEqual(len(frame), 6)

    def test_write_source_audit_emits_md_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_source_audit(tmp)
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["csv"].exists())
            md = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("Source inventory", md)
            self.assertIn("What still REQUIRES paid prop-odds data", md)
            self.assertIn("Football-Data.co.uk", md)


if __name__ == "__main__":
    unittest.main()
