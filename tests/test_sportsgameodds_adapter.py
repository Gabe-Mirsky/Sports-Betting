"""Tests for the SportsGameOdds -> player-prop-schema normalizer (offline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.player_prop_schema import validate_player_prop_snapshots  # noqa: E402
from data.prop_collection import append_snapshots, load_existing_snapshots  # noqa: E402
from data.sportsgameodds_prop_adapter import (  # noqa: E402
    american_to_decimal,
    is_player_prop_odd,
    normalize_sportsgameodds_event,
    normalize_sportsgameodds_events,
)


def _quote(odds: str, line: str, updated: str = "2026-06-11T17:24:00Z") -> dict:
    return {"odds": odds, "overUnder": line, "lastUpdatedAt": updated, "available": True}


def make_event() -> dict:
    """Synthetic SGO event: 2 players, game odds, period odds, odd corners."""

    return {
        "eventID": "evt1",
        "leagueID": "NBA",
        "sportID": "BASKETBALL",
        "status": {"startsAt": "2026-06-14T00:30:00.000Z"},
        "teams": {
            "home": {"teamID": "SAN_ANTONIO_SPURS_NBA",
                     "names": {"long": "San Antonio Spurs", "short": "SAS"}},
            "away": {"teamID": "NEW_YORK_KNICKS_NBA",
                     "names": {"long": "New York Knicks", "short": "NYK"}},
        },
        "players": {
            "JOSE_ALVARADO_1_NBA": {"playerID": "JOSE_ALVARADO_1_NBA", "name": "Jose Alvarado",
                                    "teamID": "NEW_YORK_KNICKS_NBA"},
            "DEAARON_FOX_1_NBA": {"playerID": "DEAARON_FOX_1_NBA", "name": "De'Aaron Fox",
                                  "teamID": "SAN_ANTONIO_SPURS_NBA"},
        },
        "odds": {
            # Two-sided player points market; one book at a different (alt) line,
            # one book missing the under side.
            "points-JOSE_ALVARADO_1_NBA-game-ou-over": {
                "oddID": "points-JOSE_ALVARADO_1_NBA-game-ou-over",
                "statID": "points", "statEntityID": "JOSE_ALVARADO_1_NBA",
                "periodID": "game", "betTypeID": "ou", "sideID": "over",
                "playerID": "JOSE_ALVARADO_1_NBA",
                "byBookmaker": {
                    "fanduel": _quote("+102", "4.5"),
                    "betmgm": _quote("-130", "3.5"),
                    "onesided": _quote("+100", "4.5"),
                },
            },
            "points-JOSE_ALVARADO_1_NBA-game-ou-under": {
                "oddID": "points-JOSE_ALVARADO_1_NBA-game-ou-under",
                "statID": "points", "statEntityID": "JOSE_ALVARADO_1_NBA",
                "periodID": "game", "betTypeID": "ou", "sideID": "under",
                "playerID": "JOSE_ALVARADO_1_NBA",
                "byBookmaker": {
                    "fanduel": _quote("-136", "4.5"),
                    "betmgm": _quote("-105", "3.5"),
                },
            },
            # Game (team) odds: must NEVER become player-prop rows.
            "points-away-game-ml-away": {
                "oddID": "points-away-game-ml-away",
                "statID": "points", "statEntityID": "away",
                "periodID": "game", "betTypeID": "ml", "sideID": "away",
                "byBookmaker": {"fanduel": _quote("+120", "")},
            },
            "points-all-game-ou-over": {
                "oddID": "points-all-game-ou-over",
                "statID": "points", "statEntityID": "all",
                "periodID": "game", "betTypeID": "ou", "sideID": "over",
                "byBookmaker": {"fanduel": _quote("-110", "224.5")},
            },
            # Quarter-period player odds: excluded (settles vs full-game actuals).
            "points-DEAARON_FOX_1_NBA-1q-ou-over": {
                "oddID": "points-DEAARON_FOX_1_NBA-1q-ou-over",
                "statID": "points", "statEntityID": "DEAARON_FOX_1_NBA",
                "periodID": "1q", "betTypeID": "ou", "sideID": "over",
                "playerID": "DEAARON_FOX_1_NBA",
                "byBookmaker": {"fanduel": _quote("-110", "8.5")},
            },
            # Unmapped stat: counted, not normalized.
            "firstBasket-DEAARON_FOX_1_NBA-game-ou-over": {
                "oddID": "firstBasket-DEAARON_FOX_1_NBA-game-ou-over",
                "statID": "firstBasket", "statEntityID": "DEAARON_FOX_1_NBA",
                "periodID": "game", "betTypeID": "ou", "sideID": "over",
                "playerID": "DEAARON_FOX_1_NBA",
                "byBookmaker": {"fanduel": _quote("+500", "0.5")},
            },
            # Mapped two-sided assists market.
            "assists-DEAARON_FOX_1_NBA-game-ou-over": {
                "oddID": "assists-DEAARON_FOX_1_NBA-game-ou-over",
                "statID": "assists", "statEntityID": "DEAARON_FOX_1_NBA",
                "periodID": "game", "betTypeID": "ou", "sideID": "over",
                "playerID": "DEAARON_FOX_1_NBA",
                "byBookmaker": {"draftkings": _quote("+104", "6.5")},
            },
            "assists-DEAARON_FOX_1_NBA-game-ou-under": {
                "oddID": "assists-DEAARON_FOX_1_NBA-game-ou-under",
                "statID": "assists", "statEntityID": "DEAARON_FOX_1_NBA",
                "periodID": "game", "betTypeID": "ou", "sideID": "under",
                "playerID": "DEAARON_FOX_1_NBA",
                "byBookmaker": {"draftkings": _quote("-128", "6.5")},
            },
        },
    }


class AmericanToDecimalTests(unittest.TestCase):
    def test_conversions(self) -> None:
        self.assertAlmostEqual(american_to_decimal("+100"), 2.0)
        self.assertAlmostEqual(american_to_decimal("+114"), 2.14)
        self.assertAlmostEqual(american_to_decimal("-134"), 1.746269, places=5)
        self.assertAlmostEqual(american_to_decimal("EVEN"), 2.0)
        self.assertIsNone(american_to_decimal(""))
        self.assertIsNone(american_to_decimal(None))
        self.assertIsNone(american_to_decimal("abc"))
        self.assertIsNone(american_to_decimal("0"))


class PlayerPropDetectionTests(unittest.TestCase):
    def test_game_odds_never_player_props(self) -> None:
        event = make_event()
        ml = event["odds"]["points-away-game-ml-away"]
        team_total = event["odds"]["points-all-game-ou-over"]
        self.assertFalse(is_player_prop_odd(ml, "points-away-game-ml-away"))
        self.assertFalse(is_player_prop_odd(team_total, "points-all-game-ou-over"))

    def test_player_ou_detected(self) -> None:
        event = make_event()
        over = event["odds"]["points-JOSE_ALVARADO_1_NBA-game-ou-over"]
        self.assertTrue(is_player_prop_odd(over, "points-JOSE_ALVARADO_1_NBA-game-ou-over"))

    def test_non_game_period_excluded(self) -> None:
        event = make_event()
        quarter = event["odds"]["points-DEAARON_FOX_1_NBA-1q-ou-over"]
        self.assertFalse(is_player_prop_odd(quarter, "points-DEAARON_FOX_1_NBA-1q-ou-over"))


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame, self.stats = normalize_sportsgameodds_event(
            make_event(), sport="basketball", league="NBA",
            raw_source_file="data/raw/test.json",
        )

    def test_schema_valid(self) -> None:
        result = validate_player_prop_snapshots(self.frame)
        self.assertTrue(result["valid"], result["errors"])

    def test_over_under_merged_per_book(self) -> None:
        fanduel = self.frame[
            (self.frame["bookmaker"] == "fanduel") & (self.frame["prop_type"] == "points")
        ]
        self.assertEqual(len(fanduel), 1)
        row = fanduel.iloc[0]
        self.assertEqual(row["line"], 4.5)
        self.assertAlmostEqual(row["over_price"], 2.02)
        self.assertAlmostEqual(row["under_price"], 1.735294, places=5)

    def test_alternate_line_kept_not_deleted(self) -> None:
        betmgm = self.frame[
            (self.frame["bookmaker"] == "betmgm") & (self.frame["prop_type"] == "points")
        ]
        self.assertEqual(len(betmgm), 1)
        self.assertEqual(betmgm.iloc[0]["line"], 3.5)  # different line preserved

    def test_one_sided_market_kept_with_missing_side(self) -> None:
        onesided = self.frame[self.frame["bookmaker"] == "onesided"]
        self.assertEqual(len(onesided), 1)
        row = onesided.iloc[0]
        self.assertAlmostEqual(row["over_price"], 2.0)
        self.assertTrue(pd.isna(row["under_price"]))
        self.assertGreaterEqual(self.stats["one_sided_rows"], 1)

    def test_no_game_or_period_or_unmapped_rows(self) -> None:
        self.assertGreater(self.stats["game_odds_skipped"], 0)
        self.assertGreater(self.stats["unmapped_stat_skipped"], 0)
        self.assertNotIn("firstBasket", set(self.frame["prop_type"]))
        # Only full-game player markets normalized: points (3 books) + assists (1).
        self.assertEqual(set(self.frame["prop_type"]), {"points", "assists"})

    def test_team_opponent_home_away_mapping(self) -> None:
        alvarado = self.frame[self.frame["player_name"] == "Jose Alvarado"].iloc[0]
        self.assertEqual(alvarado["team"], "NYK")
        self.assertEqual(alvarado["opponent"], "SAS")
        self.assertEqual(alvarado["home_away"], "away")
        fox = self.frame[self.frame["player_name"] == "De'Aaron Fox"].iloc[0]
        self.assertEqual(fox["team"], "SAS")
        self.assertEqual(fox["home_away"], "home")

    def test_canonical_game_key_and_metadata(self) -> None:
        row = self.frame.iloc[0]
        self.assertEqual(row["canonical_game_key"], "basketball|NBA|2026-06-13|SAS|NYK")
        self.assertEqual(row["source"], "sportsgameodds")
        self.assertEqual(row["season"], "2025-26")
        self.assertTrue(str(row["market_id"]).startswith("evt1:"))
        self.assertEqual(row["raw_source_file"], "data/raw/test.json")


class AppendDedupeTests(unittest.TestCase):
    def test_second_normalize_of_same_payload_dedupes(self) -> None:
        frame1, _ = normalize_sportsgameodds_events(
            [make_event()], sport="basketball", league="NBA", raw_source_file="a.json"
        )
        frame2, _ = normalize_sportsgameodds_events(
            [make_event()], sport="basketball", league="NBA", raw_source_file="b.json"
        )
        combined, dupes = append_snapshots(frame1, frame2)
        # raw_source_file is not a dedup key: identical market snapshots dedupe.
        self.assertEqual(len(combined), len(frame1))
        self.assertEqual(dupes, len(frame2))

    def test_load_existing_snapshots_missing_file(self) -> None:
        missing = PROJECT_ROOT / "data" / "processed" / "_does_not_exist.csv"
        frame = load_existing_snapshots(missing)
        self.assertTrue(frame.empty)


class CollectorHelperTests(unittest.TestCase):
    def test_monthly_entities_parsing(self) -> None:
        import collect_sportsgameodds_props as collector

        usage = {"rateLimits": {"per-month": {"max-entities": 2500, "current-entities": 475}}}
        remaining, cap = collector.monthly_entities(usage)
        self.assertEqual(remaining, 2025.0)
        self.assertEqual(cap, 2500.0)

    def test_monthly_entities_unlimited(self) -> None:
        import collect_sportsgameodds_props as collector

        usage = {"rateLimits": {"per-month": {"max-entities": "unlimited"}}}
        remaining, cap = collector.monthly_entities(usage)
        self.assertIsNone(remaining)
        self.assertIsNone(cap)
        self.assertEqual(collector.monthly_entities(None), (None, None))


if __name__ == "__main__":
    unittest.main()
