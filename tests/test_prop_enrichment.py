from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.prop_enrichment import (  # noqa: E402
    compute_actual_stat,
    enrich_player_prop_snapshots,
    normalize_player_name,
    run_prop_enrichment,
    settle_prop,
)


GAME_KEY = "basketball|NBA|2026-05-06|NYK|PHI"
PENDING_KEY = "basketball|NBA|2026-06-10|NYK|SAS"


def _snapshot_row(**overrides) -> dict:
    row = {
        "snapshot_time": "2026-05-06T12:00:00+00:00",
        "sport": "basketball",
        "league": "NBA",
        "season": "2025-26",
        "game_date": "2026-05-06",
        "game_start_time": "2026-05-06T23:40:00+00:00",
        "canonical_game_key": GAME_KEY,
        "player_name": "Jalen Brunson",
        "player_id": "",
        "team": "",
        "opponent": "",
        "home_away": "",
        "prop_type": "points",
        "line": 27.5,
        "over_price": 1.9,
        "under_price": 1.9,
        "bookmaker": "fanduel",
        "source": "odds_api",
        "market_id": "m1:player_points",
        "is_closing_snapshot": False,
        "minutes_to_game_start": 700.0,
        "has_result": False,
        "actual_stat_value": None,
        "over_won": None,
        "under_won": None,
        "raw_source_file": "raw.json",
    }
    row.update(overrides)
    return row


def _log_row(**overrides) -> dict:
    row = {
        "source": "nba_api",
        "source_role": "player_actuals_source",
        "sport": "basketball",
        "league": "NBA",
        "player_id": "1628973",
        "player_name": "Jalen Brunson",
        "team_id": "1610612752",
        "team_abbr": "NYK",
        "team_name": "New York Knicks",
        "opponent_abbr": "PHI",
        "is_home": True,
        "game_id": "0022500001",
        "game_date": "2026-05-06",
        "season": "2025-26",
        "season_type": "Regular Season",
        "season_start_year": 2025,
        "minutes": 36,
        "points": 32,
        "rebounds": 4,
        "assists": 7,
        "threes": 3,
        "steals": 2,
        "blocks": 1,
        "turnovers": 3,
        "canonical_game_key": GAME_KEY,
        "key_version": "v1",
    }
    row.update(overrides)
    return row


def _games_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "league": "NBA",
                "game_id": "0022500001",
                "game_date": "2026-05-06",
                "home_team_abbr": "NYK",
                "away_team_abbr": "PHI",
                "canonical_game_key": GAME_KEY,
            }
        ]
    )


def _enrich(snapshots, logs=None, games=None):
    snaps = pd.DataFrame(snapshots)
    log_frame = pd.DataFrame(logs if logs is not None else [_log_row()])
    games_frame = games if games is not None else _games_frame()
    return enrich_player_prop_snapshots(snaps, games_frame, log_frame)


class NameNormalizationTests(unittest.TestCase):
    def test_strips_punctuation_accents_and_suffixes(self) -> None:
        self.assertEqual(normalize_player_name("De'Aaron Fox"), "deaaron fox")
        self.assertEqual(normalize_player_name("Luka Dončić"), "luka doncic")
        self.assertEqual(normalize_player_name("Jaren Jackson Jr."), "jaren jackson")
        self.assertEqual(normalize_player_name("Karl-Anthony Towns"), "karl anthony towns")
        self.assertEqual(normalize_player_name(None), "")


class PlayerMatchingTests(unittest.TestCase):
    def test_matches_player_via_game_log(self) -> None:
        result = _enrich([_snapshot_row()])
        row = result["enriched"].iloc[0]
        self.assertTrue(row["player_matched"])
        self.assertEqual(row["player_id"], "1628973")
        self.assertEqual(row["player_match_method"], "game_log")
        self.assertEqual(row["team"], "NYK")
        self.assertEqual(row["opponent"], "PHI")
        self.assertEqual(row["home_away"], "home")

    def test_matches_player_with_name_variant(self) -> None:
        result = _enrich([_snapshot_row(player_name="JALEN BRUNSON")])
        self.assertTrue(result["enriched"].iloc[0]["player_matched"])

    def test_unmatched_player_reported(self) -> None:
        result = _enrich([_snapshot_row(player_name="Nonexistent Player")])
        row = result["enriched"].iloc[0]
        self.assertFalse(row["player_matched"])
        self.assertEqual(row["settlement_status"], "unmatched_player")
        self.assertFalse(row["has_result"])
        unmatched = result["unmatched_players"]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched.iloc[0]["player_name"], "Nonexistent Player")
        self.assertEqual(unmatched.iloc[0]["reason"], "no_log_history")
        self.assertEqual(result["summary"]["player_id_matched"], 0)

    def test_ambiguous_player_stays_unmatched(self) -> None:
        logs = [
            _log_row(),
            _log_row(player_id="999999", team_abbr="PHI", is_home=False, opponent_abbr="NYK"),
        ]
        snap = _snapshot_row(canonical_game_key=PENDING_KEY, game_date="2026-06-10")
        result = _enrich([snap], logs=logs)
        row = result["enriched"].iloc[0]
        self.assertFalse(row["player_matched"])
        self.assertEqual(row["settlement_status"], "ambiguous_player")
        self.assertEqual(result["unmatched_players"].iloc[0]["reason"], "ambiguous_name")


class GameMatchingTests(unittest.TestCase):
    def test_known_game_key_matches(self) -> None:
        result = _enrich([_snapshot_row()])
        self.assertTrue(result["enriched"].iloc[0]["game_matched"])
        self.assertTrue(result["unmatched_games"].empty)
        self.assertEqual(result["summary"]["game_key_match_rate"], 1.0)

    def test_unknown_game_key_reported(self) -> None:
        snap = _snapshot_row(canonical_game_key=PENDING_KEY, game_date="2026-06-10")
        result = _enrich([snap])
        self.assertFalse(result["enriched"].iloc[0]["game_matched"])
        unmatched = result["unmatched_games"]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched.iloc[0]["canonical_game_key"], PENDING_KEY)
        self.assertEqual(int(unmatched.iloc[0]["snapshots"]), 1)


class SettlementTests(unittest.TestCase):
    def _settled_row(self, prop_type: str, line: float):
        result = _enrich([_snapshot_row(prop_type=prop_type, line=line)])
        return result["enriched"].iloc[0], result["summary"]

    def test_points_settlement_over_wins(self) -> None:
        row, summary = self._settled_row("points", 27.5)  # actual 32
        self.assertTrue(row["has_result"])
        self.assertEqual(row["actual_stat_value"], 32.0)
        self.assertTrue(row["over_won"])
        self.assertFalse(row["under_won"])
        self.assertFalse(row["push"])
        self.assertEqual(row["settlement_status"], "settled")
        self.assertEqual(summary["settled"], 1)
        self.assertEqual(summary["pending"], 0)

    def test_rebounds_settlement_under_wins(self) -> None:
        row, _ = self._settled_row("rebounds", 5.5)  # actual 4
        self.assertEqual(row["actual_stat_value"], 4.0)
        self.assertFalse(row["over_won"])
        self.assertTrue(row["under_won"])
        self.assertFalse(row["push"])

    def test_assists_settlement(self) -> None:
        row, _ = self._settled_row("assists", 6.5)  # actual 7
        self.assertEqual(row["actual_stat_value"], 7.0)
        self.assertTrue(row["over_won"])

    def test_pra_settlement_sums_components(self) -> None:
        row, _ = self._settled_row("points_rebounds_assists", 40.5)  # 32+4+7=43
        self.assertEqual(row["actual_stat_value"], 43.0)
        self.assertTrue(row["over_won"])

    def test_threes_steals_blocks_turnovers_supported(self) -> None:
        log = _log_row()
        for prop_type, expected in (("threes", 3.0), ("steals", 2.0), ("blocks", 1.0), ("turnovers", 3.0)):
            self.assertEqual(compute_actual_stat(log, prop_type), expected)

    def test_push_when_actual_equals_line(self) -> None:
        row, _ = self._settled_row("points", 32.0)  # actual 32
        self.assertTrue(row["has_result"])
        self.assertTrue(row["push"])
        self.assertFalse(row["over_won"])
        self.assertFalse(row["under_won"])
        grade = settle_prop(10.0, 10.0)
        self.assertEqual(grade, {"over_won": False, "under_won": False, "push": True})

    def test_pending_game_stays_unsettled(self) -> None:
        snap = _snapshot_row(canonical_game_key=PENDING_KEY, game_date="2026-06-10")
        result = _enrich([snap])
        row = result["enriched"].iloc[0]
        self.assertTrue(row["player_matched"])  # name directory fallback
        self.assertEqual(row["player_match_method"], "name_directory")
        self.assertFalse(row["has_result"])
        self.assertEqual(row["settlement_status"], "pending_result")
        self.assertTrue(row["settlement_ready"])
        summary = result["summary"]
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["settled"], 0)
        self.assertEqual(summary["settlement_ready"], 1)

    def test_unsupported_prop_not_settled(self) -> None:
        result = _enrich([_snapshot_row(prop_type="double_double", line=0.5)])
        row = result["enriched"].iloc[0]
        self.assertFalse(row["settlement_ready"])
        self.assertEqual(row["settlement_status"], "unsupported_prop")
        self.assertFalse(row["has_result"])

    def test_non_nba_rows_pass_through(self) -> None:
        snap = _snapshot_row(league="MLB", sport="baseball", prop_type="hits", line=1.5)
        result = _enrich([snap])
        row = result["enriched"].iloc[0]
        self.assertEqual(row["settlement_status"], "not_nba")
        self.assertFalse(row["player_matched"])
        self.assertEqual(result["summary"]["nba_snapshots"], 0)
        self.assertEqual(result["summary"]["total_snapshots"], 1)


class RunnerOutputTests(unittest.TestCase):
    def test_run_prop_enrichment_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame([_snapshot_row(), _snapshot_row(player_name="Nonexistent Player")]).to_csv(
                processed / "player_prop_snapshots_normalized.csv", index=False
            )
            _games_frame().to_csv(processed / "nba_current_games_normalized.csv", index=False)
            pd.DataFrame([_log_row()]).to_csv(
                processed / "nba_current_player_game_logs_normalized.csv", index=False
            )

            summary = run_prop_enrichment(root)

            self.assertTrue((processed / "player_prop_snapshots_enriched.csv").exists())
            reports = root / "data" / "reports"
            self.assertTrue((reports / "player_prop_enrichment_summary.json").exists())
            self.assertTrue((reports / "player_prop_unmatched_players.csv").exists())
            self.assertTrue((reports / "player_prop_unmatched_games.csv").exists())

            written = json.loads((reports / "player_prop_enrichment_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["nba_snapshots"], 2)
            self.assertEqual(written["settled"], 1)
            self.assertEqual(written["player_id_matched"], 1)
            self.assertTrue(written["research_only"])
            self.assertFalse(written["approved"])
            self.assertEqual(summary["unmatched_player_names"], 1)


if __name__ == "__main__":
    unittest.main()
