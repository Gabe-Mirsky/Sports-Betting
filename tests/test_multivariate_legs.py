from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.multivariate_legs import extract_multivariate_nba_legs, summarize_multivariate_nba_legs  # noqa: E402


class TestMultivariateLegs(unittest.TestCase):
    def test_extracts_nba_spread_total_and_prop_legs_without_marking_backtestable(self) -> None:
        markets = pd.DataFrame(
            [
                {
                    "market_ticker": "KXMVE-1",
                    "market_title": "yes Boston,yes Over 221 points,yes Jayson Tatum: 25+",
                    "mve_selected_legs": str(
                        [
                            {
                                "event_ticker": "KXNBASPREAD-26MAY08NYKBOS",
                                "market_ticker": "KXNBASPREAD-26MAY08NYKBOS-BOS5",
                                "side": "yes",
                            },
                            {
                                "event_ticker": "KXNBATOTAL-26MAY08NYKBOS",
                                "market_ticker": "KXNBATOTAL-26MAY08NYKBOS-221",
                                "side": "yes",
                            },
                            {
                                "event_ticker": "KXNBAPTS-26MAY08NYKBOS",
                                "market_ticker": "KXNBAPTS-26MAY08NYKBOS-BOSJTATUM0-25",
                                "side": "yes",
                            },
                        ]
                    ),
                }
            ]
        )

        legs = extract_multivariate_nba_legs(markets)
        summary = summarize_multivariate_nba_legs(legs)

        self.assertEqual(len(legs), 3)
        self.assertEqual(summary["spread_total_leg_rows"], 2)
        self.assertEqual(summary["player_prop_leg_rows"], 1)
        self.assertEqual(summary["directly_backtestable_rows"], 0)
        self.assertTrue(summary["blocked"])
        self.assertEqual(set(legs["leg_game_date"]), {"2026-05-08"})
        self.assertEqual(set(legs["home_team_abbr"]), {"BOS"})
        self.assertEqual(set(legs["away_team_abbr"]), {"NYK"})


if __name__ == "__main__":
    unittest.main()
