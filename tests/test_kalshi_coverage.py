from __future__ import annotations

import sys
import shutil
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.coverage import build_kalshi_gap_report, build_market_truth_audit  # noqa: E402


class TestKalshiCoverage(unittest.TestCase):
    def test_gap_report_excludes_auto_matches_and_labels_preseason(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_coverage_root"
        if root.exists():
            shutil.rmtree(root)
        try:
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "game_date": "2025-10-10",
                        "home_team_abbr": "TOR",
                        "away_team_abbr": "BOS",
                        "market_ticker": "KXNBAGAME-25OCT10BOSTOR-BOS",
                        "market_title": "Boston vs Toronto Winner?",
                        "yes_team_abbr": "BOS",
                        "status": "finalized",
                        "result": "yes",
                    },
                    {
                        "game_date": "2026-05-05",
                        "home_team_abbr": "OKC",
                        "away_team_abbr": "LAL",
                        "market_ticker": "KXNBAGAME-26MAY05LALOKC-OKC",
                        "market_title": "Game 1: Los Angeles L at Oklahoma City Winner?",
                        "yes_team_abbr": "OKC",
                        "status": "finalized",
                        "result": "yes",
                    },
                ]
            ).to_csv(processed / "kalshi_possible_nba_markets.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "game_date": "2026-05-05",
                        "home_team_abbr": "OKC",
                        "away_team_abbr": "LAL",
                        "match_status": "auto_matched",
                    }
                ]
            ).to_csv(processed / "kalshi_game_market_matches.csv", index=False)

            gap_report = build_kalshi_gap_report(root)
        finally:
            if root.exists():
                shutil.rmtree(root)

        self.assertEqual(len(gap_report), 1)
        self.assertEqual(gap_report["market_ticker"].iloc[0], "KXNBAGAME-25OCT10BOSTOR-BOS")
        self.assertEqual(gap_report["gap_reason"].iloc[0], "preseason_market")

    def test_market_truth_audit_pivots_prices_and_flags_quality(self) -> None:
        root = PROJECT_ROOT / "data" / "reports" / "_test_market_truth_root"
        if root.exists():
            shutil.rmtree(root)
        try:
            processed = root / "data" / "processed"
            interim = root / "data" / "interim"
            processed.mkdir(parents=True)
            interim.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "game_id": "1",
                        "game_date": "2026-05-05",
                        "home_team_abbr": "OKC",
                        "away_team_abbr": "LAL",
                        "market_ticker": "KXNBAGAME-26MAY05LALOKC-OKC",
                        "series_ticker": "KXNBAGAME",
                        "yes_team_abbr": "OKC",
                        "match_status": "auto_matched",
                        "close_time": "2026-05-06T02:30:00Z",
                    },
                    {
                        "game_id": "2",
                        "game_date": "2026-05-06",
                        "home_team_abbr": "PHI",
                        "away_team_abbr": "NYK",
                        "market_ticker": "KXNBAGAME-26MAY12MINSAS-SAS",
                        "series_ticker": "KXNBAGAME",
                        "yes_team_abbr": "",
                        "match_status": "no_match",
                        "close_time": "2026-05-13T02:30:00Z",
                    }
                ]
            ).to_csv(processed / "kalshi_game_market_matches.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "game_id": "1",
                        "market_ticker": "KXNBAGAME-26MAY05LALOKC-OKC",
                        "snapshot_target": "pregame_60m",
                        "yes_price": 55,
                        "yes_bid": 54,
                        "yes_ask": 56,
                        "mid_price": 55,
                        "volume": 100,
                        "open_interest": 20,
                        "price_quality": "bid_ask_available",
                    },
                    {
                        "game_id": "1",
                        "market_ticker": "KXNBAGAME-26MAY05LALOKC-OKC",
                        "snapshot_target": "pregame_5m",
                        "yes_price": 58,
                        "yes_bid": 57,
                        "yes_ask": 59,
                        "mid_price": 58,
                        "volume": 125,
                        "open_interest": 24,
                        "price_quality": "bid_ask_available",
                    },
                ]
            ).to_csv(processed / "kalshi_pregame_prices.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "game_id": "1",
                        "game_date": "2026-05-05",
                        "home_team_abbr": "OKC",
                        "away_team_abbr": "LAL",
                        "game_start_time": "2026-05-06T01:30:00Z",
                    }
                ]
            ).to_csv(interim / "nba_games.csv", index=False)

            audit, summary = build_market_truth_audit(root, max_spread_cents=1.0, min_volume=10.0)
        finally:
            if root.exists():
                shutil.rmtree(root)

        self.assertEqual(len(audit), 1)
        self.assertNotIn("no_match", set(audit["match_status"]))
        self.assertEqual(audit["pregame_price_60m"].iloc[0], 55)
        self.assertEqual(audit["pregame_price_5m"].iloc[0], 58)
        self.assertEqual(audit["spread"].iloc[0], 2)
        self.assertTrue(bool(audit["wide_spread"].iloc[0]))
        self.assertEqual(summary["matched_game_markets"], 1)
        self.assertEqual(summary["usable_price_counts"]["pregame_60m"], 1)
        self.assertEqual(summary["usable_price_counts"]["pregame_30m"], 0)
        self.assertEqual(summary["ticker_mapping_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
