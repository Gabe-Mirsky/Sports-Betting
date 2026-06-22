from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.sportsbook_odds import (  # noqa: E402
    match_sportsbook_odds_to_games,
    moneyline_to_implied_probability,
    normalize_sportsbook_odds,
    select_closing_odds,
    sportsbook_match_report_by_season,
)
from data.seasons import (  # noqa: E402
    build_free_odds_split_plan,
    infer_train_bounds_from_sportsbook_coverage,
    infer_train_start_from_sportsbook_coverage,
)


class TestSportsbookOdds(unittest.TestCase):
    def test_moneyline_to_implied_probability(self) -> None:
        self.assertAlmostEqual(moneyline_to_implied_probability(-150), 150 / 250)
        self.assertAlmostEqual(moneyline_to_implied_probability(120), 100 / 220)

    def test_normalize_adds_no_vig_probabilities(self) -> None:
        odds = normalize_sportsbook_odds(
            pd.DataFrame(
                [
                    {
                        "game_date": "2024-01-01",
                        "home_team": "Boston Celtics",
                        "away_team": "New York Knicks",
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                        "sportsbook": "Example",
                        "is_closing": True,
                    }
                ]
            )
        )

        self.assertEqual(odds.loc[0, "home_team_abbr"], "BOS")
        self.assertEqual(odds.loc[0, "away_team_abbr"], "NYK")
        self.assertAlmostEqual(
            float(odds.loc[0, "home_no_vig_prob"] + odds.loc[0, "away_no_vig_prob"]),
            1.0,
        )

    def test_select_closing_prefers_flagged_latest_row(self) -> None:
        odds = normalize_sportsbook_odds(
            pd.DataFrame(
                [
                    {
                        "game_date": "2024-01-01",
                        "home_team": "BOS",
                        "away_team": "NYK",
                        "home_moneyline": -120,
                        "away_moneyline": 110,
                        "timestamp": "2024-01-01 10:00:00",
                        "is_closing": False,
                    },
                    {
                        "game_date": "2024-01-01",
                        "home_team": "BOS",
                        "away_team": "NYK",
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                        "timestamp": "2024-01-01 18:00:00",
                        "is_closing": True,
                    },
                ]
            )
        )

        selected = select_closing_odds(odds)

        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.loc[0, "home_moneyline"]), -150)

    def test_match_sportsbook_odds_to_games(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2024-01-01",
                    "season": 2023,
                    "home_team_abbr": "BOS",
                    "away_team_abbr": "NYK",
                }
            ]
        )
        odds = normalize_sportsbook_odds(
            pd.DataFrame(
                [
                    {
                        "game_date": "2024-01-01",
                        "home_team": "Boston",
                        "away_team": "Knicks",
                        "home_moneyline": -150,
                        "away_moneyline": 130,
                    }
                ]
            )
        )

        matched = match_sportsbook_odds_to_games(games, odds)

        self.assertEqual(len(matched), 1)
        self.assertTrue(pd.notna(matched.loc[0, "home_no_vig_prob"]))

    def test_fuzzy_team_aliases_match_common_sportsbook_names(self) -> None:
        odds = normalize_sportsbook_odds(
            pd.DataFrame(
                [
                    {
                        "game_date": "2025-01-01",
                        "home_team": "LA Lakers",
                        "away_team": "NY Knicks",
                        "home_moneyline": -110,
                        "away_moneyline": -105,
                    },
                    {
                        "game_date": "2025-01-02",
                        "home_team": "L.A. Clippers",
                        "away_team": "GS Warriors",
                        "home_moneyline": 120,
                        "away_moneyline": -135,
                    },
                ]
            )
        )

        self.assertEqual(odds.loc[0, "home_team_abbr"], "LAL")
        self.assertEqual(odds.loc[0, "away_team_abbr"], "NYK")
        self.assertEqual(odds.loc[1, "home_team_abbr"], "LAC")
        self.assertEqual(odds.loc[1, "away_team_abbr"], "GSW")

    def test_match_report_and_split_start_follow_sportsbook_coverage(self) -> None:
        games = pd.DataFrame(
            [
                {"game_date": "2019-01-01", "season": 2018, "home_team_abbr": "BOS", "away_team_abbr": "NYK"},
                {"game_date": "2021-01-01", "season": 2020, "home_team_abbr": "LAL", "away_team_abbr": "GSW"},
            ]
        )
        odds = normalize_sportsbook_odds(
            pd.DataFrame(
                [
                    {
                        "game_date": "2021-01-01",
                        "home_team": "LA Lakers",
                        "away_team": "GS Warriors",
                        "home_moneyline": -120,
                        "away_moneyline": 110,
                    }
                ]
            )
        )

        report = sportsbook_match_report_by_season(games, odds)

        self.assertEqual(int(report.loc[report["season"].eq(2018), "matched_games"].iloc[0]), 0)
        self.assertEqual(int(report.loc[report["season"].eq(2020), "matched_games"].iloc[0]), 1)
        self.assertEqual(
            infer_train_start_from_sportsbook_coverage(report, default_train_start_season=2018, train_end_season=2023),
            2020,
        )

    def test_train_bounds_stop_before_no_sportsbook_season(self) -> None:
        coverage = pd.DataFrame(
            [
                {"season": 2018, "matched_games": 100, "match_rate": 1.0},
                {"season": 2019, "matched_games": 100, "match_rate": 1.0},
                {"season": 2020, "matched_games": 100, "match_rate": 1.0},
                {"season": 2021, "matched_games": 100, "match_rate": 1.0},
                {"season": 2022, "matched_games": 50, "match_rate": 0.5},
                {"season": 2023, "matched_games": 0, "match_rate": 0.0},
            ]
        )

        self.assertEqual(
            infer_train_bounds_from_sportsbook_coverage(
                coverage,
                default_train_start_season=2018,
                default_train_end_season=2023,
            ),
            (2018, 2022),
        )

    def test_latest_available_split_uses_partial_latest_as_validation(self) -> None:
        coverage = pd.DataFrame(
            [
                {"season": 2018, "matched_games": 1230, "match_rate": 1.0},
                {"season": 2019, "matched_games": 1059, "match_rate": 1.0},
                {"season": 2020, "matched_games": 1080, "match_rate": 1.0},
                {"season": 2021, "matched_games": 1230, "match_rate": 1.0},
                {"season": 2022, "matched_games": 664, "match_rate": 0.54},
                {"season": 2023, "matched_games": 0, "match_rate": 0.0},
                {"season": 2024, "matched_games": 0, "match_rate": 0.0},
            ]
        )

        plan = build_free_odds_split_plan(coverage, mode="latest_available")

        self.assertEqual(plan["train_seasons"], [2018, 2019, 2020, 2021])
        self.assertEqual(plan["validation_season"], 2022)
        self.assertEqual(plan["season_splits"][2023], "outside_split")
        self.assertEqual(plan["season_splits"][2024], "outside_split")
        self.assertIn("partial free odds coverage", plan["partial_validation_warning"])

    def test_strict_full_seasons_split_uses_latest_full_season_as_validation(self) -> None:
        coverage = pd.DataFrame(
            [
                {"season": 2018, "matched_games": 1230, "match_rate": 1.0},
                {"season": 2019, "matched_games": 1059, "match_rate": 1.0},
                {"season": 2020, "matched_games": 1080, "match_rate": 1.0},
                {"season": 2021, "matched_games": 1230, "match_rate": 1.0},
                {"season": 2022, "matched_games": 664, "match_rate": 0.54},
            ]
        )

        plan = build_free_odds_split_plan(coverage, mode="strict_full_seasons")

        self.assertEqual(plan["train_seasons"], [2018, 2019, 2020])
        self.assertEqual(plan["validation_season"], 2021)
        self.assertEqual(plan["season_splits"][2018], "strict_train")
        self.assertEqual(plan["season_splits"][2021], "strict_validation")
        self.assertEqual(plan["season_splits"][2022], "outside_split")


if __name__ == "__main__":
    unittest.main()
