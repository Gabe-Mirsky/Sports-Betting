from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.player_prop_manual_review import (  # noqa: E402
    OUTPUT_FILES,
    build_manual_review_summary,
    configured_league_prop_types,
    write_manual_review_reports,
)


CONFIG = {
    "defaults": {"max_leagues_per_run": 2, "event_horizon_hours": 36},
    "leagues": {
        "NBA": {
            "sport": "basketball",
            "enabled": True,
            "modeling_priority": True,
            "collect_only": False,
            "priority": 1,
            "sources": {
                "odds_api": {
                    "sport_key": "basketball_nba",
                    "markets": {
                        "player_points": "points",
                        "player_rebounds": "rebounds",
                        "player_assists": "assists",
                        "player_threes": "threes",
                    },
                }
            },
        },
        "WNBA": {
            "sport": "basketball",
            "enabled": True,
            "modeling_priority": False,
            "collect_only": True,
            "priority": 2,
            "sources": {
                "odds_api": {"sport_key": "basketball_wnba", "markets": {"player_points": "points"}}
            },
        },
        "NHL": {
            "sport": "hockey",
            "enabled": True,
            "modeling_priority": False,
            "collect_only": True,
            "priority": 5,
            "sources": {
                "odds_api": {"sport_key": "icehockey_nhl", "markets": {"player_goals": "goals"}}
            },
        },
        "DISABLED_LEAGUE": {
            "sport": "soccer",
            "enabled": False,
            "sources": {"odds_api": {"sport_key": "soccer_x", "markets": {"player_shots": "shots"}}},
        },
    },
}


def _normalized_frame() -> pd.DataFrame:
    rows = [
        {
            "snapshot_time": "2026-06-10T01:00:00+00:00", "league": "NBA", "player_name": "Dylan Harper",
            "prop_type": "points", "line": 15.5, "over_price": 1.9, "under_price": 1.9,
            "bookmaker": "fanduel", "is_closing_snapshot": False,
        },
        {
            "snapshot_time": "2026-06-10T02:00:00+00:00", "league": "NBA", "player_name": "Dylan Harper",
            "prop_type": "rebounds", "line": 4.5, "over_price": 1.85, "under_price": "",
            "bookmaker": "draftkings", "is_closing_snapshot": True,
        },
        {
            "snapshot_time": "2026-06-10T02:30:00+00:00", "league": "NBA", "player_name": "Jose Alvarado",
            "prop_type": "points", "line": 9.5, "over_price": 2.0, "under_price": 1.8,
            "bookmaker": "", "is_closing_snapshot": False,
        },
        {
            "snapshot_time": "2026-06-10T03:00:00+00:00", "league": "WNBA", "player_name": "A Wilson",
            "prop_type": "points", "line": 22.5, "over_price": 1.9, "under_price": 1.9,
            "bookmaker": "betmgm", "is_closing_snapshot": False,
        },
    ]
    return pd.DataFrame(rows)


def _enriched_frame() -> pd.DataFrame:
    frame = _normalized_frame()
    nba = frame[frame["league"] == "NBA"].copy()
    nba["player_id"] = ["1642844", "1642844", "1630631"]
    nba["canonical_game_key"] = "basketball|NBA|2026-06-10|NYK|SAS"
    nba["player_matched"] = True
    nba["settlement_supported"] = True
    return nba


class TestManualReviewSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = build_manual_review_summary(_normalized_frame(), _enriched_frame(), CONFIG)

    def test_summary_totals_and_counts(self) -> None:
        self.assertEqual(self.summary["total_snapshots"], 4)
        self.assertEqual(self.summary["snapshots_by_league"], {"NBA": 3, "WNBA": 1})
        self.assertEqual(self.summary["snapshots_by_prop_type"]["points"], 3)
        self.assertEqual(self.summary["closing_like_snapshots"], 1)
        self.assertEqual(self.summary["latest_snapshot_time_utc"], "2026-06-10T03:00:00+00:00")
        league_prop = {
            (record["league"], record["prop_type"]): record["snapshots"]
            for record in self.summary["snapshots_by_league_prop"]
        }
        self.assertEqual(league_prop[("NBA", "points")], 2)
        self.assertEqual(league_prop[("NBA", "rebounds")], 1)
        self.assertTrue(self.summary["research_only"])
        self.assertFalse(self.summary["approved"])

    def test_configured_prop_types_missing_detection(self) -> None:
        configured = configured_league_prop_types(CONFIG)
        self.assertEqual(configured["NBA"], {"points", "rebounds", "assists", "threes"})
        self.assertNotIn("DISABLED_LEAGUE", configured)
        self.assertEqual(
            self.summary["configured_prop_types_not_collected"]["NBA"], ["assists", "threes"]
        )
        self.assertEqual(self.summary["configured_leagues_not_collected"], ["NHL"])
        reasons = "\n".join(self.summary["possible_missing_reasons"])
        self.assertIn("NHL", reasons)
        self.assertIn("max_leagues_per_run", reasons)  # NHL priority 5 > cap 2
        self.assertIn("NBA/assists", reasons)

    def test_missing_field_detection(self) -> None:
        missing = self.summary["missing_fields"]
        self.assertEqual(missing["player_name"], 0)
        self.assertEqual(missing["under_price"], 1)
        self.assertEqual(missing["bookmaker"], 1)
        nba_missing = self.summary["nba_missing_fields"]
        self.assertEqual(nba_missing["source"], "enriched")
        self.assertEqual(nba_missing["player_id"], 0)
        self.assertEqual(nba_missing["canonical_game_key"], 0)

    def test_nba_player_and_prop_summary(self) -> None:
        self.assertEqual(self.summary["nba_players"], ["Dylan Harper", "Jose Alvarado"])
        self.assertEqual(
            self.summary["nba_player_prop_types"]["Dylan Harper"], ["points", "rebounds"]
        )
        self.assertEqual(self.summary["nba_player_match_rate"], 1.0)
        self.assertTrue(self.summary["nba_usable_for_grading"])
        self.assertIn("usable", self.summary["nba_usability_verdict"].lower())

    def test_unusable_without_enrichment(self) -> None:
        summary = build_manual_review_summary(_normalized_frame(), None, CONFIG)
        self.assertFalse(summary["nba_usable_for_grading"])
        self.assertIn("enrichment run", "\n".join(summary["nba_usability_checks"]))


class TestManualReviewOutputs(unittest.TestCase):
    def test_write_manual_review_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized.csv"
            enriched = root / "enriched.csv"
            config_path = root / "prop_collection.yaml"
            _normalized_frame().to_csv(normalized, index=False)
            _enriched_frame().to_csv(enriched, index=False)
            import yaml

            config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
            reports_dir = root / "reports"

            summary = write_manual_review_reports(
                root,
                normalized_path=normalized,
                enriched_path=enriched,
                config_path=config_path,
                reports_dir=reports_dir,
            )

            for filename in OUTPUT_FILES.values():
                self.assertTrue((reports_dir / filename).exists(), filename)

            saved = json.loads((reports_dir / OUTPUT_FILES["summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["total_snapshots"], 4)

            nba_review = pd.read_csv(reports_dir / OUTPUT_FILES["nba_review"])
            self.assertEqual(len(nba_review), 3)  # 2 players x props (Harper 2, Alvarado 1)
            harper_points = nba_review[
                (nba_review["player_name"] == "Dylan Harper") & (nba_review["prop_type"] == "points")
            ]
            self.assertEqual(int(harper_points.iloc[0]["snapshots"]), 1)

            missing = pd.read_csv(reports_dir / OUTPUT_FILES["missing_fields"])
            under = missing[(missing["field"] == "under_price")]
            self.assertEqual(int(under.iloc[0]["missing_count"]), 1)

            markdown = (reports_dir / OUTPUT_FILES["summary_md"]).read_text(encoding="utf-8")
            self.assertIn("Player Prop Manual Review", markdown)
            self.assertIn("Dylan Harper", markdown)
            self.assertIn("Research-only", markdown)


if __name__ == "__main__":
    unittest.main()
