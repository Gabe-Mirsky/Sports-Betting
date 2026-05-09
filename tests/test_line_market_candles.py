from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.line_market_candles import _market_start_timestamp, _prepare_line_markets, _series_from_ticker  # noqa: E402


class TestLineMarketCandles(unittest.TestCase):
    def test_series_from_ticker_uses_prefix(self) -> None:
        self.assertEqual(_series_from_ticker("KXNBASPREAD-26MAY03TORCLE-CLE8"), "KXNBASPREAD")

    def test_market_start_prefers_occurrence_datetime(self) -> None:
        timestamp, quality = _market_start_timestamp(
            pd.Series(
                {
                    "occurrence_datetime": "2026-05-03T20:00:00Z",
                    "game_date": "2026-05-03",
                }
            )
        )

        self.assertEqual(quality, "market_occurrence_datetime")
        self.assertEqual(timestamp.isoformat(), "2026-05-03T20:00:00+00:00")

    def test_market_start_falls_back_to_estimated_7pm_eastern(self) -> None:
        timestamp, quality = _market_start_timestamp(pd.Series({"game_date": "2026-05-03"}))

        self.assertEqual(quality, "estimated_7pm_eastern")
        self.assertEqual(timestamp.isoformat(), "2026-05-03T23:00:00+00:00")

    def test_prepare_line_markets_filters_categories_and_derives_series(self) -> None:
        markets = pd.DataFrame(
            [
                {"market_ticker": "KXNBASPREAD-26MAY03TORCLE-CLE8", "occurrence_datetime": "2026-05-03T20:00:00Z"},
                {"market_ticker": "KXNBAGAME-26MAY03TORCLE-CLE", "occurrence_datetime": "2026-05-03T20:00:00Z"},
            ]
        )
        taxonomy = pd.DataFrame(
            [
                {
                    "market_ticker": "KXNBASPREAD-26MAY03TORCLE-CLE8",
                    "market_category": "spread_handicap",
                    "game_date": "2026-05-03",
                    "line_value": 8.5,
                },
                {
                    "market_ticker": "KXNBAGAME-26MAY03TORCLE-CLE",
                    "market_category": "game_winner",
                    "game_date": "2026-05-03",
                },
            ]
        )

        prepared = _prepare_line_markets(markets, taxonomy)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.loc[0, "series_ticker"], "KXNBASPREAD")


if __name__ == "__main__":
    unittest.main()
