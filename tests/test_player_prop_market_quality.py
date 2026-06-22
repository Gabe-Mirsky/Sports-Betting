from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.player_prop_market_quality import (  # noqa: E402
    OUTPUT_FILES,
    build_bookmaker_coverage,
    build_closing_coverage,
    build_line_quality,
    build_market_quality_summary,
    build_possible_alt_lines,
    write_market_quality_reports,
)
from reports.dashboard import write_static_dashboard_pages  # noqa: E402


GAME_KEY = "basketball|NBA|2026-06-10|NYK|SAS"


def _row(
    player: str,
    line: float,
    over: float | None = 1.9,
    under: float | None = 1.9,
    bookmaker: str = "fanduel",
    prop_type: str = "points",
    snapshot_time: str = "2026-06-10T01:00:00+00:00",
    closing: bool = False,
    league: str = "NBA",
    game_key: str = GAME_KEY,
) -> dict:
    return {
        "snapshot_time": snapshot_time,
        "league": league,
        "game_date": "2026-06-10",
        "canonical_game_key": game_key,
        "player_name": player,
        "prop_type": prop_type,
        "line": line,
        "over_price": over,
        "under_price": under,
        "bookmaker": bookmaker,
        "is_closing_snapshot": closing,
    }


def _snapshots_frame() -> pd.DataFrame:
    rows = [
        # Clean market: one line, two snapshots, latest is closing-like.
        _row("Jalen Brunson", 27.5, 1.85, 1.95, snapshot_time="2026-06-10T01:00:00+00:00"),
        _row("Jalen Brunson", 27.5, 1.88, 1.92, snapshot_time="2026-06-10T22:00:00+00:00", closing=True),
        # Alt-line ladder: many lines at the same snapshot time, wide range,
        # no closing snapshot; main line is the price-balanced 20.5.
        *[
            _row("Victor Wembanyama", line, over, under, bookmaker="bovada")
            for line, over, under in [
                (14.5, 1.30, 3.40), (16.5, 1.45, 2.70), (18.5, 1.65, 2.20),
                (20.5, 1.90, 1.92), (22.5, 2.30, 1.60), (24.5, 2.90, 1.40),
            ]
        ],
        # Line movement: two lines at different times, most frequent wins.
        _row("Josh Hart", 7.5, snapshot_time="2026-06-10T01:00:00+00:00", prop_type="rebounds"),
        _row("Josh Hart", 8.5, snapshot_time="2026-06-10T12:00:00+00:00", prop_type="rebounds"),
        _row("Josh Hart", 8.5, snapshot_time="2026-06-10T18:00:00+00:00", prop_type="rebounds"),
        # Small same-time ladder without two-sided prices anywhere: cannot be
        # balanced-price resolved; wide range -> uncertain.
        _row("Karl-Anthony Towns", 10.5, 1.4, None, bookmaker="betrivers"),
        _row("Karl-Anthony Towns", 15.5, 2.1, None, bookmaker="betrivers"),
        _row("Karl-Anthony Towns", 20.5, 4.0, None, bookmaker="betrivers"),
        # Missing both prices plus suspicious price/line values.
        _row("Miles McBride", 11.5, None, None, bookmaker="betmgm"),
        _row("Miles McBride", 11.5, 250.0, 1.0, bookmaker="betmgm", snapshot_time="2026-06-10T02:00:00+00:00"),
        _row("Mitchell Robinson", -2.5, 1.9, 1.9, bookmaker="betmgm"),
        # Exact duplicate snapshot rows.
        _row("OG Anunoby", 16.5, 1.9, 1.9, bookmaker="draftkings"),
        _row("OG Anunoby", 16.5, 1.9, 1.9, bookmaker="draftkings"),
        # Missing bookmaker / player / game key.
        _row("Landry Shamet", 9.5, bookmaker=""),
        _row("", 5.5, bookmaker="fanduel"),
        _row("Luke Kornet", 3.5, game_key=""),
        # Second league + book for coverage/overlap checks.
        _row("A Wilson", 22.5, league="WNBA", bookmaker="betmgm",
             snapshot_time="2026-06-10T22:00:00+00:00", closing=True),
        _row("A Wilson", 22.5, league="WNBA", bookmaker="fanduel"),
    ]
    return pd.DataFrame(rows)


def _market(line_quality: pd.DataFrame, player: str, bookmaker: str | None = None) -> pd.Series:
    rows = line_quality[line_quality["player_name"] == player]
    if bookmaker:
        rows = rows[rows["bookmaker"] == bookmaker]
    assert len(rows) == 1, f"expected one market for {player}, got {len(rows)}"
    return rows.iloc[0]


class TestLineQuality(unittest.TestCase):
    def setUp(self) -> None:
        self.snaps = _snapshots_frame()
        self.line_quality = build_line_quality(self.snaps)

    def test_clean_market_uses_closing_line(self) -> None:
        market = _market(self.line_quality, "Jalen Brunson")
        self.assertEqual(market["line_quality_label"], "clean")
        self.assertEqual(market["main_line_reason"], "closing_snapshot")
        self.assertEqual(market["likely_main_line"], 27.5)
        self.assertEqual(market["closing_line"], 27.5)
        self.assertTrue(market["has_closing_snapshot"])
        self.assertFalse(market["possible_alt_lines"])
        self.assertFalse(market["low_snapshot_count"])
        self.assertEqual(market["flags"], "")

    def test_alt_line_ladder_flagged_and_resolved_by_balanced_prices(self) -> None:
        market = _market(self.line_quality, "Victor Wembanyama")
        self.assertTrue(market["possible_alt_lines"])
        self.assertTrue(market["wide_line_range"])
        self.assertEqual(market["unique_lines"], 6)
        self.assertEqual(market["max_lines_same_snapshot_time"], 6)
        self.assertEqual(market["likely_main_line"], 20.5)
        self.assertEqual(market["main_line_reason"], "balanced_prices")
        self.assertEqual(market["line_quality_label"], "main_plus_alt_lines")
        self.assertEqual(market["min_line"], 14.5)
        self.assertEqual(market["max_line"], 24.5)
        self.assertIn("14.5", market["likely_alt_lines"])
        self.assertNotIn("20.5", market["likely_alt_lines"].split("|"))

    def test_line_movement_uses_most_frequent_line(self) -> None:
        market = _market(self.line_quality, "Josh Hart")
        self.assertEqual(market["line_quality_label"], "line_movement")
        self.assertEqual(market["main_line_reason"], "most_frequent")
        self.assertEqual(market["likely_main_line"], 8.5)
        self.assertEqual(market["most_common_line"], 8.5)
        self.assertEqual(market["latest_line"], 8.5)
        self.assertFalse(market["possible_alt_lines"])
        self.assertFalse(market["wide_line_range"])

    def test_wide_unresolvable_ladder_marked_uncertain(self) -> None:
        market = _market(self.line_quality, "Karl-Anthony Towns")
        self.assertTrue(market["wide_line_range"])
        self.assertTrue(market["possible_alt_lines"])
        self.assertEqual(market["line_quality_label"], "uncertain")
        self.assertEqual(market["main_line_reason"], "most_frequent")
        self.assertFalse(market["both_prices_present"])

    def test_missing_and_suspicious_price_detection(self) -> None:
        market = _market(self.line_quality, "Miles McBride")
        self.assertTrue(market["missing_prices"])
        self.assertEqual(market["rows_missing_both_prices"], 1)
        # over 250.0 above max and under 1.0 below min both count.
        self.assertTrue(market["suspicious_price_values"])
        self.assertEqual(market["suspicious_price_count"], 1)

    def test_suspicious_line_values(self) -> None:
        market = _market(self.line_quality, "Mitchell Robinson")
        self.assertTrue(market["suspicious_line_values"])
        self.assertIn("suspicious_line_values", market["flags"])

    def test_duplicate_exact_snapshots(self) -> None:
        market = _market(self.line_quality, "OG Anunoby")
        self.assertTrue(market["duplicate_exact_snapshots"])
        self.assertEqual(market["duplicate_snapshot_rows"], 1)

    def test_missing_key_fields_flagged(self) -> None:
        self.assertTrue(_market(self.line_quality, "Landry Shamet")["missing_bookmaker"])
        self.assertTrue(_market(self.line_quality, "(missing)")["missing_player"])
        self.assertTrue(_market(self.line_quality, "Luke Kornet")["missing_game_key"])

    def test_low_snapshot_count(self) -> None:
        market = _market(self.line_quality, "Landry Shamet")
        self.assertTrue(market["low_snapshot_count"])

    def test_empty_frame(self) -> None:
        empty = build_line_quality(pd.DataFrame())
        self.assertTrue(empty.empty)


class TestAltLinesOutput(unittest.TestCase):
    def test_alt_lines_listed_without_main(self) -> None:
        snaps = _snapshots_frame()
        line_quality = build_line_quality(snaps)
        alts = build_possible_alt_lines(snaps, line_quality)
        wemby = alts[alts["player_name"] == "Victor Wembanyama"]
        self.assertEqual(len(wemby), 5)  # 6 lines minus the main 20.5
        self.assertNotIn(20.5, wemby["likely_alt_line"].tolist())
        self.assertTrue((wemby["likely_main_line"] == 20.5).all())
        # Markets without the possible_alt_lines flag are not listed.
        self.assertNotIn("Josh Hart", alts["player_name"].tolist())


class TestBookmakerCoverage(unittest.TestCase):
    def test_coverage_overlap_and_missing_books(self) -> None:
        snaps = _snapshots_frame()
        coverage, summary = build_bookmaker_coverage(snaps)

        nba_fanduel = coverage[(coverage["league"] == "NBA") & (coverage["bookmaker"] == "fanduel")]
        self.assertEqual(int(nba_fanduel.iloc[0]["players"]), 4)  # Brunson, Hart, Kornet, (missing)
        self.assertEqual(int(nba_fanduel.iloc[0]["prop_types"]), 2)

        # WNBA: A Wilson points quoted by two books -> overlap of 2.
        self.assertEqual(
            summary["bookmaker_overlap_by_league"]["WNBA"], {"markets_with_2_books": 1}
        )
        # Books seen in other leagues but absent from WNBA.
        for book in ("bovada", "betrivers", "draftkings"):
            self.assertIn(book, summary["missing_books_by_league"]["WNBA"])

        best = summary["nba_best_bookmakers"]
        self.assertTrue(best)
        self.assertEqual(best[0]["bookmaker"], "fanduel")  # most distinct NBA markets (4)
        markets = [book["markets"] for book in best]
        self.assertEqual(markets, sorted(markets, reverse=True))


class TestClosingCoverage(unittest.TestCase):
    def test_closing_counts_and_clv_verdict(self) -> None:
        snaps = _snapshots_frame()
        line_quality = build_line_quality(snaps)
        coverage, summary = build_closing_coverage(snaps, line_quality)

        self.assertEqual(summary["total_closing_snapshots"], 2)
        self.assertEqual(summary["closing_by_league"], {"NBA": 1, "WNBA": 1})
        self.assertEqual(summary["closing_by_prop_type"], {"points": 2})
        self.assertEqual(summary["closing_by_bookmaker"], {"fanduel": 1, "betmgm": 1})
        self.assertEqual(
            summary["markets_without_closing"],
            len(line_quality) - 2,
        )
        # Only 1 of many NBA markets has a closing snapshot -> not CLV-ready.
        self.assertFalse(summary["nba_clv_ready"])
        self.assertIn("closing", summary["clv_readiness_verdict"].lower())

        row = coverage[
            (coverage["league"] == "NBA")
            & (coverage["prop_type"] == "points")
            & (coverage["bookmaker"] == "fanduel")
        ].iloc[0]
        self.assertEqual(int(row["closing_snapshots"]), 1)
        self.assertEqual(int(row["markets_with_closing"]), 1)


class TestSummaryAndOutputs(unittest.TestCase):
    def test_summary_counts(self) -> None:
        snaps = _snapshots_frame()
        line_quality = build_line_quality(snaps)
        _, book_summary = build_bookmaker_coverage(snaps)
        _, closing_summary = build_closing_coverage(snaps, line_quality)
        summary = build_market_quality_summary(snaps, line_quality, book_summary, closing_summary)

        self.assertTrue(summary["research_only"])
        self.assertFalse(summary["approved"])
        self.assertEqual(summary["total_snapshots"], len(snaps))
        self.assertEqual(summary["total_markets_audited"], len(line_quality))
        self.assertEqual(summary["possible_alt_line_markets"], int(line_quality["possible_alt_lines"].sum()))
        self.assertEqual(summary["wide_line_range_markets"], int(line_quality["wide_line_range"].sum()))
        self.assertEqual(summary["missing_price_markets"], int(line_quality["missing_prices"].sum()))
        self.assertIn("flag_counts", summary)
        # Uncertain + suspicious-line markets exist, so NBA is not clean here.
        self.assertFalse(summary["nba_clean_enough_for_modeling"])

    def test_write_reports_and_dashboard_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized.csv"
            _snapshots_frame().to_csv(normalized, index=False)
            reports_dir = root / "reports"

            summary = write_market_quality_reports(
                root, normalized_path=normalized, reports_dir=reports_dir
            )

            for filename in OUTPUT_FILES.values():
                self.assertTrue((reports_dir / filename).exists(), filename)

            saved = json.loads(
                (reports_dir / OUTPUT_FILES["summary_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved["total_markets_audited"], summary["total_markets_audited"])
            self.assertEqual(saved["inputs"]["source_file"], str(normalized))

            markdown = (reports_dir / OUTPUT_FILES["summary_md"]).read_text(encoding="utf-8")
            self.assertIn("Player Prop Market Quality Audit", markdown)
            self.assertIn("Research-only", markdown)
            self.assertIn("possible_alt_lines", markdown)

            main_lines = pd.read_csv(reports_dir / OUTPUT_FILES["likely_main_lines"])
            self.assertIn("likely_main_line", main_lines.columns)
            self.assertIn("main_line_reason", main_lines.columns)

            # Dashboard player-props page stays simple and links advanced reports.
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            page_html = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", page_html)
            self.assertIn("Advanced reports", page_html)
            self.assertIn("player_prop_line_quality.csv", page_html)
            self.assertIn("player_prop_possible_alt_lines.csv", page_html)
            self.assertIn("player_prop_market_quality.md", page_html)
            self.assertNotIn("Markets audited", page_html)
            self.assertNotIn("Best-Covered NBA Bookmakers", page_html)

    def test_dashboard_section_empty_without_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp)
            written = write_static_dashboard_pages(reports_dir, reports_dir)
            page = next(p for p in written if p.name == "player_props.html")
            page_html = page.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", page_html)
            self.assertIn("No qualifying research bets yet.", page_html)
            self.assertNotIn("No market quality audit yet", page_html)


if __name__ == "__main__":
    unittest.main()
