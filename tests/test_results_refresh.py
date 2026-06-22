from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.prop_settlement_refresh import (  # noqa: E402
    NEWLY_SETTLED_FILENAME,
    REFRESH_SUMMARY_FILENAME,
    run_results_refresh,
)
from reports.dashboard import write_static_dashboard_pages  # noqa: E402


PLAYED_KEY = "basketball|NBA|2026-05-06|NYK|PHI"
NEXT_KEY = "basketball|NBA|2026-06-09|NYK|SAS"
FUTURE_KEY = "basketball|NBA|2026-07-01|NYK|SAS"


def _team_log_rows(game_id: str, date: str, home: str, away: str,
                   home_pts: int, away_pts: int) -> list[dict]:
    shared = {"GAME_DATE": date, "nba_season": "2025-26"}
    return [
        {"GAME_ID": game_id, "MATCHUP": f"{home} vs. {away}", "TEAM_ABBREVIATION": home,
         "TEAM_ID": 1, "TEAM_NAME": home, "PTS": home_pts,
         "WL": "W" if home_pts > away_pts else "L", **shared},
        {"GAME_ID": game_id, "MATCHUP": f"{away} @ {home}", "TEAM_ABBREVIATION": away,
         "TEAM_ID": 2, "TEAM_NAME": away, "PTS": away_pts,
         "WL": "W" if away_pts > home_pts else "L", **shared},
    ]


def _player_log_row(game_id: str, date: str, team: str, opponent: str, is_home: bool,
                    player_id: int, player_name: str, points: int, rebounds: int = 4,
                    assists: int = 7) -> dict:
    matchup = f"{team} vs. {opponent}" if is_home else f"{team} @ {opponent}"
    return {
        "PLAYER_ID": player_id, "PLAYER_NAME": player_name, "TEAM_ID": 1,
        "TEAM_ABBREVIATION": team, "TEAM_NAME": team, "MATCHUP": matchup,
        "GAME_ID": game_id, "GAME_DATE": date, "MIN": 36, "PTS": points,
        "REB": rebounds, "AST": assists, "FG3M": 3, "STL": 2, "BLK": 1, "TOV": 3,
        "nba_season": "2025-26",
    }


def _snapshot_row(player_name: str, game_key: str, game_date: str,
                  prop_type: str = "points", line: float = 27.5) -> dict:
    return {
        "snapshot_time": f"{game_date}T12:00:00+00:00",
        "sport": "basketball",
        "league": "NBA",
        "season": "2025-26",
        "game_date": game_date,
        "game_start_time": f"{game_date}T23:40:00+00:00",
        "canonical_game_key": game_key,
        "player_name": player_name,
        "player_id": "",
        "team": "",
        "opponent": "",
        "home_away": "",
        "prop_type": prop_type,
        "line": line,
        "over_price": 1.9,
        "under_price": 1.9,
        "bookmaker": "fanduel",
        "source": "odds_api",
        "market_id": f"m:{player_name}:{prop_type}:{line}",
        "is_closing_snapshot": False,
        "minutes_to_game_start": 700.0,
        "has_result": False,
        "actual_stat_value": None,
        "over_won": None,
        "under_won": None,
        "raw_source_file": "raw.json",
    }


def _played_game_caches() -> tuple[list[dict], list[dict]]:
    """Raw cache rows for the played 2026-05-06 NYK vs PHI game."""

    team_rows = _team_log_rows("001", "2026-05-06", "NYK", "PHI", 110, 100)
    player_rows = [
        _player_log_row("001", "2026-05-06", "NYK", "PHI", True, 1628973,
                        "Jalen Brunson", points=32, rebounds=4),
        _player_log_row("001", "2026-05-06", "NYK", "PHI", True, 1628404,
                        "Josh Hart", points=14, rebounds=11),
    ]
    return team_rows, player_rows


def _write_caches(root: Path, team_rows: list[dict], player_rows: list[dict]) -> None:
    player_dir = root / "data" / "raw" / "nba" / "player"
    player_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(team_rows).to_csv(root / "data" / "raw" / "nba" / "league_game_log_2025.csv", index=False)
    pd.DataFrame(player_rows).to_csv(player_dir / "player_game_log_2025.csv", index=False)


def _write_snapshots(root: Path, snapshots: list[dict]) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(snapshots).to_csv(processed / "player_prop_snapshots_normalized.csv", index=False)


def _fail_downloader(*args, **kwargs) -> None:
    raise AssertionError("downloader must not be called in cache-only mode")


class CacheOnlyRefreshTests(unittest.TestCase):
    def test_cache_only_refresh_settles_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_caches(root, *_played_game_caches())
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", PLAYED_KEY, "2026-05-06", "points", 27.5),
                _snapshot_row("Jalen Brunson", FUTURE_KEY, "2026-07-01", "points", 30.5),
            ])

            summary = run_results_refresh(root, download=False, downloader=_fail_downloader)

            self.assertEqual(summary["mode"], "cache_only")
            self.assertEqual(summary["actuals_import"]["status"], "ok")
            settlement = summary["settlement"]
            self.assertEqual(settlement["pending_before_refresh"], 0)
            self.assertEqual(settlement["settled_total"], 1)
            self.assertEqual(settlement["newly_settled"], 1)
            self.assertEqual(settlement["pending_after_refresh"], 1)
            self.assertTrue(summary["research_only"])
            self.assertFalse(summary["approved"])

            processed = root / "data" / "processed"
            reports = root / "data" / "reports"
            for path in (
                processed / "player_prop_snapshots_enriched.csv",
                processed / "nba_current_games_normalized.csv",
                processed / "nba_current_player_game_logs_normalized.csv",
                reports / "player_prop_enrichment_summary.json",
                reports / "player_prop_unmatched_players.csv",
                reports / "player_prop_unmatched_games.csv",
                reports / REFRESH_SUMMARY_FILENAME,
                reports / NEWLY_SETTLED_FILENAME,
            ):
                self.assertTrue(path.exists(), f"missing output: {path}")

            written = json.loads((reports / REFRESH_SUMMARY_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(written["settlement"], settlement)

    def test_missing_raw_caches_skip_actuals_import(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", FUTURE_KEY, "2026-07-01", "points", 30.5),
            ])
            # Pre-existing normalized actuals must survive a cache-less refresh.
            processed = root / "data" / "processed"
            games_path = processed / "nba_current_games_normalized.csv"
            games_path.write_text("sentinel", encoding="utf-8")

            summary = run_results_refresh(root, download=False)

            self.assertEqual(summary["actuals_import"]["status"], "skipped_no_raw_caches")
            self.assertEqual(games_path.read_text(encoding="utf-8"), "sentinel")


class DownloadModeTests(unittest.TestCase):
    def test_download_mode_uses_mocked_downloader_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", PLAYED_KEY, "2026-05-06", "points", 27.5),
            ])
            calls: list[dict] = []

            def fake_downloader(raw_dir, min_season, max_season) -> None:
                calls.append({"raw_dir": Path(raw_dir), "min_season": min_season,
                              "max_season": max_season})
                _write_caches(root, *_played_game_caches())

            summary = run_results_refresh(
                root, download=True, min_season=2024, max_season=2025,
                downloader=fake_downloader,
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["raw_dir"], root / "data" / "raw")
            self.assertEqual(calls[0]["min_season"], 2024)
            self.assertEqual(calls[0]["max_season"], 2025)
            self.assertEqual(summary["mode"], "download")
            # The mocked download happened before import, so the prop settled.
            self.assertEqual(summary["settlement"]["settled_total"], 1)
            self.assertEqual(summary["actuals_import"]["status"], "ok")


class SettlementTransitionTests(unittest.TestCase):
    def test_pending_prop_settles_once_results_arrive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            team_rows, player_rows = _played_game_caches()
            _write_caches(root, team_rows, player_rows)
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", NEXT_KEY, "2026-06-09", "points", 23.5),
            ])

            first = run_results_refresh(root)
            self.assertEqual(first["settlement"]["pending_after_refresh"], 1)
            self.assertEqual(first["settlement"]["newly_settled"], 0)

            # The 2026-06-09 game completes: results land in the raw caches.
            team_rows += _team_log_rows("002", "2026-06-09", "NYK", "SAS", 120, 111)
            player_rows.append(
                _player_log_row("002", "2026-06-09", "NYK", "SAS", True, 1628973,
                                "Jalen Brunson", points=25)
            )
            _write_caches(root, team_rows, player_rows)

            second = run_results_refresh(root)
            settlement = second["settlement"]
            self.assertEqual(settlement["pending_before_refresh"], 1)
            self.assertEqual(settlement["pending_after_refresh"], 0)
            self.assertEqual(settlement["newly_settled"], 1)
            self.assertEqual(settlement["still_pending"], 0)
            self.assertEqual(settlement["newly_settled_by_prop_type"], {"points": 1})
            self.assertEqual(settlement["newly_settled_by_game"], {NEXT_KEY: 1})
            self.assertEqual(settlement["unsettled_games"], [])

            newly = pd.read_csv(root / "data" / "reports" / NEWLY_SETTLED_FILENAME)
            self.assertEqual(len(newly), 1)
            self.assertEqual(newly.iloc[0]["player_name"], "Jalen Brunson")
            self.assertEqual(float(newly.iloc[0]["actual_stat_value"]), 25.0)
            self.assertTrue(bool(newly.iloc[0]["over_won"]))

            enriched = pd.read_csv(root / "data" / "processed" / "player_prop_snapshots_enriched.csv")
            self.assertEqual(enriched.iloc[0]["settlement_status"], "settled")

    def test_future_game_stays_pending_across_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_caches(root, *_played_game_caches())
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", FUTURE_KEY, "2026-07-01", "points", 30.5),
                _snapshot_row("Josh Hart", FUTURE_KEY, "2026-07-01", "rebounds", 8.5),
            ])

            run_results_refresh(root)
            second = run_results_refresh(root)

            settlement = second["settlement"]
            self.assertEqual(settlement["pending_before_refresh"], 2)
            self.assertEqual(settlement["pending_after_refresh"], 2)
            self.assertEqual(settlement["still_pending"], 2)
            self.assertEqual(settlement["newly_settled"], 0)
            self.assertEqual(len(settlement["unsettled_games"]), 1)
            game = settlement["unsettled_games"][0]
            self.assertEqual(game["canonical_game_key"], FUTURE_KEY)
            self.assertEqual(game["pending_snapshots"], 2)
            self.assertEqual(game["players"], 2)


class SummaryCountTests(unittest.TestCase):
    def test_settlement_summary_counts_and_match_rates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_caches(root, *_played_game_caches())
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", PLAYED_KEY, "2026-05-06", "points", 27.5),
                _snapshot_row("Josh Hart", PLAYED_KEY, "2026-05-06", "points", 15.5),
                _snapshot_row("Josh Hart", PLAYED_KEY, "2026-05-06", "rebounds", 9.5),
                _snapshot_row("Jalen Brunson", FUTURE_KEY, "2026-07-01", "points", 30.5),
            ])

            summary = run_results_refresh(root)

            settlement = summary["settlement"]
            self.assertEqual(settlement["settled_total"], 3)
            self.assertEqual(settlement["pending_after_refresh"], 1)
            self.assertEqual(settlement["newly_settled"], 3)
            self.assertEqual(settlement["settled_by_prop_type"], {"points": 2, "rebounds": 1})
            self.assertEqual(settlement["settled_by_game"], {PLAYED_KEY: 3})
            enrichment = summary["enrichment"]
            self.assertEqual(enrichment["nba_snapshots"], 4)
            self.assertEqual(enrichment["player_match_rate"], 1.0)
            self.assertEqual(enrichment["game_match_rate"], 0.75)  # future game has no actuals

    def test_dashboard_shows_settlement_refresh_section(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_caches(root, *_played_game_caches())
            _write_snapshots(root, [
                _snapshot_row("Jalen Brunson", PLAYED_KEY, "2026-05-06", "points", 27.5),
                _snapshot_row("Josh Hart", FUTURE_KEY, "2026-07-01", "rebounds", 8.5),
            ])
            run_results_refresh(root)

            reports_dir = root / "data" / "reports"
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            html = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", html)
            self.assertIn("Advanced reports", html)
            self.assertIn("player_prop_settlement_refresh_summary.json", html)
            self.assertIn("Games Collected", html)
            self.assertNotIn("Newly Settled Props", html)
            self.assertNotIn("Jalen Brunson", html)


if __name__ == "__main__":
    unittest.main()
