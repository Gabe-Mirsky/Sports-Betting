from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.adapters import ADAPTER_REGISTRY, get_adapter  # noqa: E402
from data.adapters.odds_api_adapter import flatten_odds_api_events  # noqa: E402
from data.source_adapter import (  # noqa: E402
    ADAPTER_METHODS,
    ENTITY_SCHEMAS,
    UnsupportedCapabilityError,
    coerce_to_project_schema,
)
from data.source_audit import (  # noqa: E402
    build_audit_summary,
    build_field_coverage_table,
    build_source_audit_table,
    render_markdown_report,
)
from data.source_catalog import NBA_PLAYER_PROP_FIELDS, SOURCE_CATALOG, get_source


class TestSourceCatalog(unittest.TestCase):
    def test_catalog_keys_unique_and_registered(self) -> None:
        keys = [source.key for source in SOURCE_CATALOG]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), set(ADAPTER_REGISTRY))

    def test_declared_capabilities_are_known_methods(self) -> None:
        for source in SOURCE_CATALOG:
            self.assertTrue(source.adapter_capabilities.issubset(set(ADAPTER_METHODS)))

    def test_available_fields_are_in_master_list(self) -> None:
        for source in SOURCE_CATALOG:
            self.assertEqual(source.unknown_field_names(), (), f"{source.key} declares unknown fields")

    def test_missing_fields_are_complement_of_available(self) -> None:
        kalshi = get_source("kalshi")
        self.assertIn("player_id", kalshi.missing_required_fields())
        self.assertNotIn("prop_line", kalshi.missing_required_fields())


class TestAuditReport(unittest.TestCase):
    def test_audit_table_has_one_row_per_source(self) -> None:
        table = build_source_audit_table()
        self.assertEqual(len(table), len(SOURCE_CATALOG))
        for column in ["supports_player_props", "supports_historical_odds", "supports_closing_prices", "rate_limits", "priority"]:
            self.assertIn(column, table.columns)

    def test_field_coverage_counts_player_prop_sources(self) -> None:
        coverage = build_field_coverage_table()
        self.assertEqual(len(coverage), len(NBA_PLAYER_PROP_FIELDS))
        prop_line = coverage[coverage["field"].eq("prop_line")].iloc[0]
        self.assertTrue(prop_line["kalshi"])
        self.assertTrue(prop_line["odds_api"])
        self.assertFalse(prop_line["nba_api"])

    def test_summary_lists_capability_sources(self) -> None:
        summary = build_audit_summary()
        self.assertEqual(summary["sources_reviewed"], len(SOURCE_CATALOG))
        self.assertIn("kalshi", summary["player_prop_sources"])
        self.assertIn("odds_api", summary["player_prop_sources"])
        self.assertNotIn("nba_api", summary["player_prop_sources"])
        self.assertFalse(summary["approved"])

    def test_markdown_report_renders_sections(self) -> None:
        report = render_markdown_report()
        self.assertIn("# NBA Player-Prop Data Source Audit", report)
        self.assertIn("Capability Matrix", report)
        self.assertIn("Field Coverage", report)


class TestAdapterInterface(unittest.TestCase):
    def test_every_adapter_instantiates_and_maps_to_catalog(self) -> None:
        for key in ADAPTER_REGISTRY:
            adapter = get_adapter(key)
            self.assertEqual(adapter.capability.key, key)

    def test_unsupported_method_raises(self) -> None:
        nba = get_adapter("nba_api")
        with self.assertRaises(UnsupportedCapabilityError):
            nba.fetch_market_odds()

    def test_coerce_to_project_schema_orders_and_fills(self) -> None:
        frame = pd.DataFrame([{"points": 30, "extra_unused": 1}])
        out = coerce_to_project_schema(frame, "player_game_logs", source="test", sport="basketball", league="NBA")
        self.assertEqual(list(out.columns), ENTITY_SCHEMAS["player_game_logs"])
        self.assertEqual(out.loc[0, "points"], 30)
        self.assertEqual(out.loc[0, "source"], "test")
        self.assertNotIn("extra_unused", out.columns)


class TestNbaApiAdapterNormalize(unittest.TestCase):
    def test_player_log_normalization(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "PLAYER_ID": 201939,
                    "PLAYER_NAME": "Stephen Curry",
                    "TEAM_ABBREVIATION": "GSW",
                    "GAME_ID": "0022500001",
                    "GAME_DATE": "2025-10-25",
                    "MIN": 34,
                    "PTS": 30,
                    "REB": 5,
                    "AST": 6,
                    "FG3M": 7,
                    "BLK": 0,
                    "STL": 2,
                    "TOV": 3,
                    "MATCHUP": "GSW vs. LAL",
                    "season_start_year": 2025,
                }
            ]
        )
        out = get_adapter("nba_api").normalize_to_project_schema(raw, "player_game_logs")
        self.assertEqual(list(out.columns), ENTITY_SCHEMAS["player_game_logs"])
        self.assertTrue(bool(out.loc[0, "is_home"]))
        self.assertEqual(out.loc[0, "opponent_abbr"], "LAL")
        self.assertEqual(out.loc[0, "points"], 30)
        self.assertEqual(out.loc[0, "source"], "nba_api")


class TestOddsApiAdapterNormalize(unittest.TestCase):
    def _sample_events(self) -> list[dict]:
        return [
            {
                "id": "evt1",
                "commence_time": "2025-10-25T23:00:00Z",
                "home_team": "GSW",
                "away_team": "LAL",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "last_update": "2025-10-25T22:00:00Z",
                        "markets": [
                            {
                                "key": "player_points",
                                "outcomes": [
                                    {"name": "Over", "description": "Stephen Curry", "point": 27.5, "price": 1.91},
                                    {"name": "Under", "description": "Stephen Curry", "point": 27.5, "price": 1.91},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

    def test_flatten_and_normalize(self) -> None:
        flat = flatten_odds_api_events(self._sample_events())
        self.assertEqual(len(flat), 2)
        odds = get_adapter("odds_api").normalize_to_project_schema(flat, "market_odds")
        self.assertEqual(list(odds.columns), ENTITY_SCHEMAS["market_odds"])
        over = odds[odds["side"].eq("over")].iloc[0]
        self.assertEqual(over["stat_type"], "points")
        self.assertEqual(over["line"], 27.5)
        self.assertAlmostEqual(float(over["implied_prob"]), 1.0 / 1.91, places=4)

    def test_fetch_without_key_raises(self) -> None:
        adapter = get_adapter("odds_api", config={"api_key": None})
        adapter.api_key = None
        with self.assertRaises(RuntimeError):
            adapter.fetch_market_odds()


class TestKaggleCsvAdapter(unittest.TestCase):
    def test_alias_normalization(self) -> None:
        raw = pd.DataFrame([{"Player": "Nikola Jokic", "Tm": "DEN", "Date": "2025-11-01", "PTS": 28, "TRB": 12, "AST": 9}])
        out = get_adapter("kaggle_csv").normalize_to_project_schema(raw, "player_game_logs")
        self.assertEqual(out.loc[0, "player_name"], "Nikola Jokic")
        self.assertEqual(out.loc[0, "team_abbr"], "DEN")
        self.assertEqual(out.loc[0, "rebounds"], 12)
        self.assertEqual(out.loc[0, "source"], "kaggle_csv")


if __name__ == "__main__":
    unittest.main()
