from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.basketball_join_readiness import (  # noqa: E402
    build_join_readiness,
    render_player_prop_path_md,
)
from data.ehallmar_importer import (  # noqa: E402
    import_ehallmar,
    normalize_game_odds,
    normalize_games,
    normalize_player_game_logs,
)
from data.zachht_importer import normalize_basketball_odds  # noqa: E402


def _write(folder: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(folder / name, index=False)


def _ehallmar_fixture(folder: Path) -> None:
    _write(
        folder,
        "nba_teams_all.csv",
        pd.DataFrame({"league_id": [0, 0], "team_id": [1, 2], "min_year": [1946, 1948], "max_year": [2018, 2018], "abbreviation": ["BOS", "LAL"]}),
    )
    _write(
        folder,
        "nba_games_all.csv",
        pd.DataFrame(
            {
                "game_id": [100, 100],
                "game_date": ["2018-01-05", "2018-01-05"],
                "matchup": ["BOS vs. LAL", "LAL @ BOS"],
                "team_id": [1, 2],
                "is_home": ["t", "f"],
                "wl": ["W", "L"],
                "pts": [110, 104],
                "a_team_id": [2, 1],
                "season_year": [2017, 2017],
                "season_type": ["Regular Season", "Regular Season"],
                "season": ["2017-18", "2017-18"],
            }
        ),
    )
    _write(
        folder,
        "nba_players_game_stats.csv",
        pd.DataFrame(
            {
                "player_id": [50, 51],
                "player_name": ["Jayson Tatum", "LeBron James"],
                "team_abbreviation": ["BOS", "LAL"],
                "game_id": [100, 100],
                "game_date": ["2018-01-05", "2018-01-05"],
                "matchup": ["BOS vs. LAL", "LAL @ BOS"],
                "min": [35, 38],
                "pts": [27, 31],
                "reb": [8, 9],
                "ast": [5, 11],
                "fg3m": [3, 2],
                "blk": [1, 0],
                "stl": [2, 1],
                "tov": [3, 4],
                "season_year": [2017, 2017],
                "season": ["2017-18", "2017-18"],
            }
        ),
    )
    _write(
        folder,
        "nba_betting_money_line.csv",
        pd.DataFrame({"game_id": [100], "book_name": ["Pinnacle"], "book_id": [238], "team_id": [1], "a_team_id": [2], "price1": [-150.0], "price2": [130.0]}),
    )
    _write(
        folder,
        "nba_betting_spread.csv",
        pd.DataFrame({"game_id": [100], "book_name": ["Pinnacle"], "book_id": [238], "team_id": [1], "a_team_id": [2], "spread1": [-3.5], "spread2": [3.5], "price1": [-110.0], "price2": [-110.0]}),
    )
    _write(
        folder,
        "nba_betting_totals.csv",
        pd.DataFrame({"game_id": [100], "book_name": ["Pinnacle"], "book_id": [238], "team_id": [1], "a_team_id": [2], "total1": [220.5], "total2": [220.5], "price1": [-105.0], "price2": [-115.0]}),
    )


class TestEhallmarImporter(unittest.TestCase):
    def test_games_pivot_home_away_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _ehallmar_fixture(folder)
            games = normalize_games(folder)
            self.assertEqual(len(games), 1)
            row = games.iloc[0]
            self.assertEqual(row["home_team_abbr"], "BOS")
            self.assertEqual(row["away_team_abbr"], "LAL")
            self.assertEqual(row["home_score"], 110)
            self.assertEqual(row["away_score"], 104)
            self.assertTrue(bool(row["home_win"]))
            self.assertEqual(row["winner"], "home")
            self.assertEqual(row["source_role"], "player_actuals_source")

    def test_player_logs_matchup_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _ehallmar_fixture(folder)
            logs = normalize_player_game_logs(folder)
            self.assertEqual(len(logs), 2)
            tatum = logs[logs["player_name"].eq("Jayson Tatum")].iloc[0]
            self.assertTrue(bool(tatum["is_home"]))
            self.assertEqual(tatum["opponent_abbr"], "LAL")
            self.assertEqual(tatum["points"], 27)
            self.assertEqual(tatum["threes"], 3)
            lebron = logs[logs["player_name"].eq("LeBron James")].iloc[0]
            self.assertFalse(bool(lebron["is_home"]))
            self.assertEqual(lebron["opponent_abbr"], "BOS")

    def test_game_odds_long_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _ehallmar_fixture(folder)
            odds = normalize_game_odds(folder)
            self.assertEqual(set(odds["market_type"]), {"moneyline", "spread", "total"})
            self.assertTrue((odds["market_scope"] == "game_market").all())
            ml = odds[odds["market_type"].eq("moneyline")].iloc[0]
            self.assertEqual(ml["team_abbr"], "BOS")
            self.assertEqual(ml["opponent_abbr"], "LAL")
            self.assertEqual(ml["side1_price"], -150.0)

    def test_import_writes_outputs_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _ehallmar_fixture(folder)
            processed = folder / "processed"
            reports = folder / "reports"
            summary = import_ehallmar(folder, processed, reports)
            self.assertEqual(summary["source_role"], "player_actuals_source")
            self.assertFalse(summary["has_player_prop_lines"])
            self.assertTrue((processed / "nba_games_normalized.csv").exists())
            self.assertTrue((processed / "nba_player_game_logs_normalized.csv").exists())
            self.assertTrue((processed / "nba_game_odds_normalized.csv").exists())
            self.assertTrue((reports / "ehallmar_import_summary.json").exists())


def _zachht_fixture(folder: Path) -> None:
    _write(
        folder,
        "nba_main_lines.csv",
        pd.DataFrame(
            {
                "team1": ["Boston Celtics", "Boston Celtics", "Miami Heat"],
                "team2": ["Los Angeles Lakers", "Los Angeles Lakers", "Orlando Magic"],
                "game_link": [
                    "https://www.pinnacle.com/en/basketball/nba/bos-vs-lal/999/",
                    "https://www.pinnacle.com/en/basketball/nba/bos-vs-lal/999/",
                    "https://www.pinnacle.com/en/basketball/nba/mia-vs-orl/1000/",
                ],
                "team1_moneyline": [1.9, 1.85, 1.5],
                "team2_moneyline": [2.0, 2.05, 2.6],
                "team1_spread": [-2.5, -2.5, -5.5],
                "team1_spread_odds": [1.91, 1.92, 1.9],
                "team2_spread": [2.5, 2.5, 5.5],
                "team2_spread_odds": [1.91, 1.9, 1.9],
                "over_total": [221.5, 222.0, 210.0],
                "over_total_odds": [1.9, 1.9, 1.91],
                "under_total": [221.5, 222.0, 210.0],
                "under_total_odds": [1.9, 1.9, 1.9],
                "timestamp": ["2025-10-01 18:00:00", "2025-10-01 23:00:00", "2025-10-02 19:00:00"],
            }
        ),
    )


class TestZachhtImporter(unittest.TestCase):
    def test_normalize_and_open_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _zachht_fixture(folder)
            snaps = normalize_basketball_odds(folder)
            self.assertEqual(len(snaps), 3)
            self.assertTrue((snaps["market_type"] == "game_market").all())
            self.assertEqual(set(snaps["sportsbook"]), {"Pinnacle"})
            self.assertEqual(snaps.loc[snaps["team1_abbr"].eq("BOS"), "league"].iloc[0], "NBA")
            game999 = snaps[snaps["game_ref"].eq("999")].sort_values("snapshot_time")
            self.assertEqual(list(game999["n_snapshots_for_game"]), [2, 2])
            self.assertTrue(bool(game999.iloc[0]["is_opening_snapshot"]))
            self.assertTrue(bool(game999.iloc[-1]["is_closing_snapshot"]))
            self.assertFalse(bool(game999.iloc[0]["is_closing_snapshot"]))
            self.assertEqual(snaps.loc[0, "odds_format"], "decimal")


class TestJoinReadiness(unittest.TestCase):
    def _write_normalized(self, processed: Path) -> None:
        _write(processed, "nba_games_normalized.csv", pd.DataFrame({
            "game_id": [100, 101], "game_date": ["2018-01-05", "2018-01-06"],
            "home_team_abbr": ["BOS", "MIA"], "away_team_abbr": ["LAL", "ORL"],
            "home_score": [110, 99], "away_score": [104, 101],
        }))
        _write(processed, "nba_game_odds_normalized.csv", pd.DataFrame({
            "game_id": [100], "market_type": ["moneyline"], "book_name": ["Pinnacle"],
        }))
        _write(processed, "nba_player_game_logs_normalized.csv", pd.DataFrame({
            "player_id": [50, 51], "player_name": ["A", "B"], "game_id": [100, 101],
            "game_date": ["2018-01-05", "2018-01-06"], "team_abbr": ["BOS", "MIA"],
        }))
        _write(processed, "basketball_odds_snapshots_normalized.csv", pd.DataFrame({
            "league": ["NBA"], "game_ref": ["999"], "team1_abbr": ["BOS"], "team2_abbr": ["LAL"],
            "snapshot_time": ["2025-10-01 18:00:00"], "is_opening_snapshot": [True], "is_closing_snapshot": [True],
        }))

    def test_join_readiness_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp)
            self._write_normalized(processed)
            report = build_join_readiness(processed)
            self.assertEqual(report["sources"]["ehallmar"]["nba_games"], 2)
            self.assertTrue(report["id_joins"]["within_ehallmar_player_logs_to_games"]["works"])
            self.assertEqual(report["id_joins"]["within_ehallmar_player_logs_to_games"]["matched_game_ids"], 2)
            self.assertEqual(report["id_joins"]["within_ehallmar_games_to_odds"]["games_with_odds"], 1)
            self.assertFalse(report["id_joins"]["cross_source_ehallmar_gameid_vs_zachht_gameref"]["works"])
            self.assertEqual(report["date_team_join"]["overlapping_game_keys"], 0)
            self.assertEqual(report["missing"]["games_without_any_odds"], 1)
            self.assertFalse(report["player_props_available"])

    def test_prop_path_md_has_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp)
            self._write_normalized(processed)
            report = build_join_readiness(processed)
            md = render_player_prop_path_md(report)
            self.assertIn("What we now have", md)
            self.assertIn("What we still do NOT have", md)
            self.assertIn("Recommended schema", md)
            self.assertIn("stat_type", md)


if __name__ == "__main__":
    unittest.main()
