from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.kalshi_taxonomy import build_market_taxonomy  # noqa: E402


class TestKalshiTaxonomy(unittest.TestCase):
    def test_kxnbagame_winner_market_is_classified_with_ticker_teams(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBAGAME-26MAY07LALOKC-LAL",
                    "market_title": "Game 2: Los Angeles L at Oklahoma City Winner?",
                    "yes_sub_title": "Los Angeles L",
                    "rules_primary": "If Los Angeles L wins the professional basketball game, then Yes.",
                }
            ]
        )

        taxonomy = build_market_taxonomy(markets)
        row = taxonomy.iloc[0]

        self.assertEqual(row["market_category"], "game_winner")
        self.assertEqual(row["home_team_abbr"], "OKC")
        self.assertEqual(row["away_team_abbr"], "LAL")
        self.assertEqual(row["yes_team_abbr"], "LAL")
        self.assertGreaterEqual(row["taxonomy_confidence"], 0.9)

    def test_core_non_winner_market_types_are_classified(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "SPREAD-1",
                    "market_title": "Will Boston cover the -4.5 spread against New York?",
                    "rules_primary": "",
                },
                {
                    "market_ticker": "TOTAL-1",
                    "market_title": "Will Boston vs New York total points go over 221.5?",
                    "rules_primary": "",
                },
                {
                    "market_ticker": "TEAMTOTAL-1",
                    "market_title": "Will Boston score over 112.5 points in the game?",
                    "rules_primary": "",
                },
                {
                    "market_ticker": "PLAYER-1",
                    "market_title": "Will Jayson Tatum score over 27.5 points?",
                    "rules_primary": "",
                },
                {
                    "market_ticker": "SERIES-1",
                    "market_title": "Will Boston win the series against New York?",
                    "rules_primary": "",
                },
            ]
        )

        taxonomy = build_market_taxonomy(markets)
        by_ticker = taxonomy.set_index("market_ticker")

        self.assertEqual(by_ticker.loc["SPREAD-1", "market_category"], "spread_handicap")
        self.assertAlmostEqual(by_ticker.loc["SPREAD-1", "line_value"], -4.5)
        self.assertEqual(by_ticker.loc["TOTAL-1", "market_category"], "total_points_over_under")
        self.assertEqual(by_ticker.loc["TOTAL-1", "direction"], "over")
        self.assertAlmostEqual(by_ticker.loc["TOTAL-1", "line_value"], 221.5)
        self.assertEqual(by_ticker.loc["TEAMTOTAL-1", "market_category"], "team_total")
        self.assertEqual(by_ticker.loc["PLAYER-1", "market_category"], "player_points_rebounds_assists")
        self.assertEqual(by_ticker.loc["PLAYER-1", "stat_type"], "points")
        self.assertEqual(by_ticker.loc["SERIES-1", "market_category"], "series_playoff")

    def test_unrecognized_market_is_weird_ambiguous(self) -> None:
        taxonomy = build_market_taxonomy(
            pd.DataFrame([{"market_ticker": "ODD-1", "market_title": "Will something strange happen?"}])
        )

        self.assertEqual(taxonomy.loc[0, "market_category"], "weird_ambiguous")
        self.assertLess(taxonomy.loc[0, "taxonomy_confidence"], 0.7)

    def test_multivariate_combo_market_is_not_treated_as_single_game_bet(self) -> None:
        taxonomy = build_market_taxonomy(
            pd.DataFrame(
                [
                    {
                        "market_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026ABC-123",
                        "market_title": "yes New York wins by over 4.5 points,yes Over 212.5 points scored",
                        "mve_collection_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-R",
                    }
                ]
            )
        )

        row = taxonomy.iloc[0]
        self.assertEqual(row["market_category"], "weird_ambiguous")
        self.assertEqual(row["market_scope"], "multivariate")
        self.assertIn("multivariate combination market", row["taxonomy_notes"])

    def test_missing_multivariate_fields_do_not_override_winner_ticker(self) -> None:
        taxonomy = build_market_taxonomy(
            pd.DataFrame(
                [
                    {
                        "market_ticker": "KXNBAGAME-26MAY07LALOKC-LAL",
                        "market_title": "Los Angeles Lakers vs Oklahoma City Thunder Winner?",
                        "mve_collection_ticker": pd.NA,
                    }
                ]
            )
        )

        self.assertEqual(taxonomy.loc[0, "market_category"], "game_winner")

    def test_direct_underlying_spread_total_and_player_props_are_classified(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBASPREAD-26MAY03TORCLE-CLE8",
                    "market_title": "Cleveland wins by over 8.5 points?",
                    "yes_sub_title": "Cleveland",
                },
                {
                    "market_ticker": "KXNBATOTAL-26MAY03TORCLE-211",
                    "market_title": "Game 7: Toronto at Cleveland: Total Points",
                    "yes_sub_title": "Over 211.5 points scored",
                    "rules_primary": "If the teams collectively score more than 211.5 points, then Yes.",
                },
                {
                    "market_ticker": "KXNBA3PT-26MAY05CLEDET-CLEDMITCHELL45-1",
                    "market_title": "Donovan Mitchell: 1+ threes",
                    "rules_primary": "If Donovan Mitchell records 1+ Three Pointers, then Yes.",
                },
            ]
        )

        taxonomy = build_market_taxonomy(markets)
        by_ticker = taxonomy.set_index("market_ticker")

        spread = by_ticker.loc["KXNBASPREAD-26MAY03TORCLE-CLE8"]
        self.assertEqual(spread["market_category"], "spread_handicap")
        self.assertEqual(spread["away_team_abbr"], "TOR")
        self.assertEqual(spread["home_team_abbr"], "CLE")
        self.assertEqual(spread["yes_team_abbr"], "CLE")
        self.assertAlmostEqual(spread["line_value"], 8.5)

        total = by_ticker.loc["KXNBATOTAL-26MAY03TORCLE-211"]
        self.assertEqual(total["market_category"], "total_points_over_under")
        self.assertEqual(total["direction"], "over")
        self.assertAlmostEqual(total["line_value"], 211.5)

        player = by_ticker.loc["KXNBA3PT-26MAY05CLEDET-CLEDMITCHELL45-1"]
        self.assertEqual(player["market_category"], "player_points_rebounds_assists")
        self.assertEqual(player["stat_type"], "three_pointers")
        self.assertEqual(player["player_name"], "Donovan Mitchell")
        self.assertAlmostEqual(player["line_value"], 1.0)
        self.assertEqual(player["direction"], "over")

    def test_newly_discovered_kxnba_series_are_classified_by_ticker(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBATEAMTOTAL-26MAY03TORCLE-CLE112",
                    "market_title": "Game 7: Toronto at Cleveland: Team Total",
                    "yes_sub_title": "Cleveland over 112.5 points",
                },
                {
                    "market_ticker": "KXNBA1HSPREAD-26MAY03TORCLE-CLE4",
                    "market_title": "Toronto vs Cleveland: First Half Spread",
                    "yes_sub_title": "Cleveland -4.5",
                },
                {
                    "market_ticker": "KXNBA1HWINNER-26MAY03TORCLE-CLE",
                    "market_title": "Toronto vs Cleveland: First Half Winner",
                    "yes_sub_title": "Cleveland",
                },
                {
                    "market_ticker": "KXNBA2D-26MAY03TORCLE-PLAYER",
                    "market_title": "Evan Mobley: Double Double",
                },
                {
                    "market_ticker": "KXNBASERIESSCORE-26MAY03TORCLE-CLE",
                    "market_title": "Series Exact Score: Toronto vs Cleveland",
                },
            ]
        )

        taxonomy = build_market_taxonomy(markets).set_index("market_ticker")

        self.assertEqual(taxonomy.loc["KXNBATEAMTOTAL-26MAY03TORCLE-CLE112", "market_category"], "team_total")
        self.assertEqual(taxonomy.loc["KXNBA1HSPREAD-26MAY03TORCLE-CLE4", "stat_type"], "first_half_spread")
        self.assertEqual(taxonomy.loc["KXNBA1HWINNER-26MAY03TORCLE-CLE", "stat_type"], "first_half_winner")
        self.assertEqual(taxonomy.loc["KXNBA2D-26MAY03TORCLE-PLAYER", "stat_type"], "double_double")
        self.assertEqual(taxonomy.loc["KXNBASERIESSCORE-26MAY03TORCLE-CLE", "market_category"], "series_playoff")


if __name__ == "__main__":
    unittest.main()
