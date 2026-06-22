from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reports.dashboard import (  # noqa: E402
    STATIC_DASHBOARD_PAGES,
    _build_matchup_predictions_page,
    _build_recorded_games_tab,
    _build_team_availability_page,
    _recorded_games_frame,
    _render_recorded_games_section,
    build_dashboard_html,
    format_local_datetime,
    write_dashboard,
)


def _panel_html(page_html: str, league_code: str) -> str:
    """Return just the HTML of one league panel (from its marker to the next panel)."""
    marker = f'id="league-{league_code}"'
    start = page_html.index(marker)
    rest = page_html[start + len(marker):]
    nxt = rest.find('id="league-')
    return rest if nxt == -1 else rest[:nxt]


class TestMatchupPredictionsPage(unittest.TestCase):
    def test_table_shows_simplified_columns_and_parlay_creator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "id": "soccer_international_2026-06-20_japan_tunisia_friendly",
                        "sport": "soccer",
                        "league": "international",
                        "game_date": "2026-06-20T00:00:00",
                        "team_a": "Japan",
                        "team_b": "Tunisia",
                        "prob_team_a_win": 0.655,
                        "prob_draw": 0.238,
                        "prob_team_b_win": 0.107,
                        "predicted_outcome": "Japan win",
                        "confidence_level": "Medium",
                        "confidence_score": 0.417,
                        "data_quality": "usable",
                        "key_reasons": "Japan has better recent form.; Japan has more rest.",
                        "main_risks": "International friendlies can rotate lineups.",
                        "data_quality_warnings": "Friendly match; lineups may be experimental.",
                        "model_version": "matchup_baseline_v1",
                    }
                ]
            ).to_csv(reports / "matchup_predictions_today.csv", index=False)
            (reports / "team_availability_validation.json").write_text(
                """{
                  "overall_status": "WARNING",
                  "coverage": {
                    "total_fixture_teams": 2,
                    "fixture_teams_with_availability": 1,
                    "fixture_teams_missing_availability": 1,
                    "coverage_percentage": 50.0,
                    "missing_teams": ["Tunisia"]
                  },
                  "injury_data": {"stale_rows_older_than_48h": 0},
                  "warnings": ["No availability rows for 1 fixture team(s): ['Tunisia']."],
                  "issues": [],
                  "team_rows": []
                }""",
                encoding="utf-8",
            )

            out = _build_matchup_predictions_page(reports)

        self.assertIn("These are model-implied probabilities, not sportsbook odds.", out)
        self.assertIn("Availability coverage:", out)
        self.assertIn("1 of 2 fixture teams", out)
        self.assertIn("Tunisia", out)
        self.assertIn("Download CSV", out)
        self.assertIn("65.5%", out)
        self.assertIn("23.8%", out)
        self.assertIn("10.7%", out)

        # Simplified, fits-on-screen column headers (no horizontal scroll table).
        for column in ["Date", "Sport", "League", "Teams", "Prediction", "Reasoning"]:
            self.assertIn(f">{column}</th>", out)

        # Teams are combined into one column; predicted outcome and reasoning stay visible.
        self.assertIn("Japan", out)
        self.assertIn("Japan win", out)
        self.assertIn("Japan has better recent form.", out)
        self.assertIn("International friendlies can rotate lineups.", out)
        self.assertIn("Friendly match; lineups may be experimental.", out)

        # Parlay creator is present with selectable legs.
        self.assertIn("Parlay creator", out)
        self.assertIn("mp-pick-box", out)
        self.assertIn("data-pick-prob", out)
        self.assertIn("Combined model probability", out)
        self.assertIn("Fair decimal odds", out)
        self.assertIn("id='parlay-legs'", out)


class TestTeamAvailabilityPage(unittest.TestCase):
    def test_team_availability_page_renders_coverage_and_team_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            (reports / "team_availability_validation.json").write_text(
                """{
                  "overall_status": "WARNING",
                  "coverage": {
                    "total_fixture_teams": 2,
                    "fixture_teams_with_availability": 1,
                    "fixture_teams_missing_availability": 1,
                    "coverage_percentage": 50.0,
                    "missing_teams": ["Tunisia"]
                  },
                  "injury_data": {
                    "stale_rows_older_than_48h": 1,
                    "status_counts": {
                      "out": 1,
                      "doubtful": 0,
                      "questionable": 1,
                      "probable": 0,
                      "available": 0,
                      "unknown": 0
                    }
                  },
                  "warnings": ["No availability rows for 1 fixture team(s): ['Tunisia']."],
                  "issues": [],
                  "team_rows": [
                    {
                      "team": "Japan",
                      "has_availability_data": true,
                      "players_listed": 2,
                      "key_players_out": 1,
                      "questionable_players": 1,
                      "stale_data_warning": true,
                      "last_updated": "2026-06-18T00:00:00",
                      "source": "manual",
                      "notes": "checked local report"
                    },
                    {
                      "team": "Tunisia",
                      "has_availability_data": false,
                      "players_listed": 0,
                      "key_players_out": 0,
                      "questionable_players": 0,
                      "stale_data_warning": false,
                      "last_updated": "",
                      "source": "",
                      "notes": ""
                    }
                  ]
                }""",
                encoding="utf-8",
            )

            out = _build_team_availability_page(reports)

        self.assertIn("Team Availability", out)
        self.assertIn("Team availability is a data-quality input, not betting information.", out)
        self.assertIn("Coverage", out)
        self.assertIn("50.0%", out)
        self.assertIn("Japan", out)
        self.assertIn("Tunisia", out)
        self.assertIn("checked local report", out)


class TestSportsMarketResearchDashboard(unittest.TestCase):
    def test_dashboard_html_is_simple_league_dashboard(self) -> None:
        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")

        self.assertIn("Sports Market Research Dashboard", html)
        self.assertIn("Research-only. No approved bets are live.", html)
        for label in ["NBA", "MLB", "WNBA", "NHL", "World Cup"]:
            self.assertIn(f">{label}</button>", html)

        for card in ["Games Collected", "Complete Markets", "Research Bets", "Last Data Collection"]:
            self.assertGreaterEqual(html.count(card), 5)

        self.assertIn("Games / Markets Summary", html)
        self.assertIn("Research Bets We Would Place", html)
        self.assertIn("No qualifying research bets yet.", html)
        self.assertIn("approved=false", html)
        self.assertIn("team_availability.html", html)

    def test_last_collection_time_uses_12_hour_new_york_format(self) -> None:
        self.assertEqual(
            format_local_datetime("2026-06-15T00:05:00+00:00"),
            "2026-06-14 08:05 PM",
        )

        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")
        self.assertRegex(html, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} (AM|PM)")
        self.assertIn("America/New_York", html)

    def test_missing_reports_do_not_crash_and_show_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            reports.mkdir()
            html = build_dashboard_html(reports)

        self.assertIn("Sports Market Research Dashboard", html)
        self.assertIn("No qualifying research bets yet.", html)
        self.assertIn("Waiting for data", html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn("KeyError", html)

    def test_old_excessive_tabs_are_not_in_main_nav(self) -> None:
        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")

        for old_marker in [
            'data-tab="home"',
            'data-tab="model-readiness"',
            'data-tab="betting-paper"',
            'data-tab="logs-health"',
            "Command Center",
            "Legacy Player Prop Detail Sections",
            "Player Props Research Dashboard",
        ]:
            self.assertNotIn(old_marker, html)

        main_nav = re.search(r'<nav class="league-tabs"[^>]*>(.*?)</nav>', html, re.S)
        self.assertIsNotNone(main_nav)
        nav_html = main_nav.group(1)
        self.assertIn("NBA", nav_html)
        self.assertIn("World Cup", nav_html)
        self.assertNotIn("Parlay", nav_html)
        self.assertNotIn("Model", nav_html)

    def test_no_raw_json_block_is_visible(self) -> None:
        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")

        self.assertNotIn("<pre>", html)
        self.assertNotIn("<pre>{", html)
        self.assertNotIn("</pre>", html)
        self.assertNotIn("raw json", html.lower())

    def test_write_dashboard_outputs_player_props_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "player_props.html"
            write_dashboard(PROJECT_ROOT / "data" / "reports", output)

            self.assertTrue(output.exists())
            html = output.read_text(encoding="utf-8")
            self.assertIn("Sports Market Research Dashboard", html)
            self.assertIn("No qualifying research bets yet.", html)

            player_props = output.parent / "player_props.html"
            self.assertTrue(player_props.exists())
            self.assertIn("Sports Market Research Dashboard", player_props.read_text(encoding="utf-8"))

            report = PROJECT_ROOT / "data" / "reports" / "dashboard_simplification_report.md"
            self.assertTrue(report.exists())
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("Dashboard Simplification Report", report_text)
            self.assertIn("League counts", report_text)

            for page in STATIC_DASHBOARD_PAGES:
                if page == "dashboard.html":
                    continue
                self.assertTrue((output.parent / page).exists(), page)


class TestRecordedGamesTab(unittest.TestCase):
    # Canonical key layout: sport|league|game_date|home_team|away_team
    K_SETTLED = "basketball|NBA|2026-06-10|NYK|SAS"
    K_CLOSING = "basketball|NBA|2026-06-12|LAL|BOS"
    K_ODDS = "basketball|NBA|2026-06-13|MIA|DEN"
    K_OUTCOME = "basketball|NBA|2026-06-01|GSW|PHX"
    K_SCHED = "basketball|NBA|2026-06-20|CHI|MIL"

    def _write_fixtures(self, root: Path) -> Path:
        processed = root / "processed"
        reports = root / "reports"
        processed.mkdir(parents=True)
        reports.mkdir(parents=True)

        # Odds snapshots: settled (2), closing (1), plain odds (1, far from tip).
        pd.DataFrame(
            [
                {"snapshot_time": "2026-06-10T22:30:00Z", "canonical_game_key": self.K_SETTLED,
                 "is_closing_snapshot": True, "minutes_to_game_start": 30, "bookmaker": "fanduel", "prop_type": "points"},
                {"snapshot_time": "2026-06-10T18:00:00Z", "canonical_game_key": self.K_SETTLED,
                 "is_closing_snapshot": False, "minutes_to_game_start": 300, "bookmaker": "draftkings", "prop_type": "rebounds"},
                {"snapshot_time": "2026-06-12T23:15:00Z", "canonical_game_key": self.K_CLOSING,
                 "is_closing_snapshot": True, "minutes_to_game_start": 45, "bookmaker": "fanduel", "prop_type": "points"},
                {"snapshot_time": "2026-06-13T15:00:00Z", "canonical_game_key": self.K_ODDS,
                 "is_closing_snapshot": False, "minutes_to_game_start": 200, "bookmaker": "betmgm", "prop_type": "assists"},
            ]
        ).to_csv(processed / "player_prop_snapshots_normalized.csv", index=False)

        # Enriched settlement state: settled game graded; odds game has a pending prop.
        pd.DataFrame(
            [
                {"canonical_game_key": self.K_SETTLED, "settlement_status": "settled", "settlement_supported": True},
                {"canonical_game_key": self.K_SETTLED, "settlement_status": "settled", "settlement_supported": True},
                {"canonical_game_key": self.K_CLOSING, "settlement_status": "unsupported_prop", "settlement_supported": False},
                {"canonical_game_key": self.K_ODDS, "settlement_status": "pending", "settlement_supported": True},
            ]
        ).to_csv(processed / "player_prop_snapshots_enriched.csv", index=False)

        # Final scores -> outcome available for the settled and outcome-only games.
        pd.DataFrame(
            [
                {"canonical_game_key": self.K_SETTLED, "home_score": 110, "away_score": 100},
                {"canonical_game_key": self.K_OUTCOME, "home_score": 120, "away_score": 115},
            ]
        ).to_csv(processed / "nba_current_games_normalized.csv", index=False)

        pd.DataFrame(
            [{"canonical_game_key": self.K_SETTLED, "outcome": "over_won"}]
        ).to_csv(reports / "player_prop_settlement_outcomes.csv", index=False)

        # Schedule-only game (no odds, no outcome).
        pd.DataFrame(
            [{"game_id": self.K_SCHED, "league": "NBA", "away_team": "MIL", "home_team": "CHI"}]
        ).to_csv(reports / "upcoming_games.csv", index=False)
        return reports

    def test_classifies_all_five_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = self._write_fixtures(Path(tmp))
            out = _build_recorded_games_tab(reports)

        # Section + the exact required explanation text.
        self.assertIn("Recorded Games", out)
        self.assertIn("Recorded does not always mean odds were recorded.", out)
        self.assertIn("outcomes/stats without historical odds", out)

        # All 16 columns are present as table headers.
        for column in [
            "league", "game_date", "away_team", "home_team", "canonical_game_key",
            "outcome_available", "game_odds_available", "odds_snapshot_count", "first_snapshot_time",
            "latest_snapshot_time", "has_closing_snapshot", "sportsbook_count",
            "prop_market_count", "settled_prop_count", "pending_prop_count", "status",
        ]:
            self.assertIn(f"<th>{column}</th>", out)

        # Exactly one game in each prop/outcome status bucket.
        self.assertIn("Games tracked: 5", out)
        for status in ["settled", "closing_recorded", "prop_odds_recorded", "outcome_only", "scheduled_only"]:
            self.assertIn(f"{status}: 1", out)

        # Keys are rendered.
        self.assertIn(self.K_SETTLED, out)
        self.assertIn(self.K_SCHED, out)

    def test_pending_prop_counted_for_ungraded_supported_prop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = self._write_fixtures(Path(tmp))
            # The odds-only game (DEN @ MIA) has one supported-but-unsettled prop.
            row = re.search(
                r"<tr>(?:(?!</tr>).)*" + re.escape(self.K_ODDS) + r".*?</tr>",
                _build_recorded_games_tab(reports),
                re.S,
            )
        self.assertIsNotNone(row)
        cells = re.findall(r"<td>(.*?)</td>", row.group(0))
        # Column order places settled_prop_count, pending_prop_count, status last three.
        self.assertEqual(cells[-3], "0")  # settled_prop_count
        self.assertEqual(cells[-2], "1")  # pending_prop_count
        self.assertEqual(cells[-1], "prop_odds_recorded")

    def test_empty_inputs_render_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            (Path(tmp) / "processed").mkdir(parents=True)
            reports.mkdir(parents=True)
            out = _build_recorded_games_tab(reports)
        self.assertIn("No recorded games yet", out)
        self.assertNotIn("Traceback", out)

    def test_section_is_wired_into_live_dashboard(self) -> None:
        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")
        self.assertIn("Recorded Games", html)
        self.assertIn("Recorded does not always mean odds were recorded.", html)

    def test_includes_historical_backfill_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            reports = Path(tmp) / "reports"
            processed.mkdir(parents=True)
            reports.mkdir(parents=True)
            # A free historical game with game odds only (no props, no outcome).
            pd.DataFrame(
                [{
                    "canonical_game_key": "soccer|EPL|2020-08-01|ARSENAL|CHELSEA",
                    "league": "EPL", "game_date": "2020-08-01",
                    "home_team": "ARSENAL", "away_team": "CHELSEA", "source": "football_data",
                    "outcome_available": True, "game_odds_available": True,
                    "prop_odds_available": False, "closing_available": False,
                }]
            ).to_csv(processed / "historical_game_inventory.csv", index=False)
            out = _build_recorded_games_tab(reports)

        self.assertIn("soccer|EPL|2020-08-01|ARSENAL|CHELSEA", out)
        self.assertIn("<th>game_odds_available</th>", out)
        # A game-odds-only game classifies as game_odds_recorded.
        self.assertIn("game_odds_recorded: 1", out)
        self.assertIn("Games tracked: 1", out)

    def test_historical_backfill_tab_surfaces_untabbed_leagues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            reports = Path(tmp) / "reports"
            processed.mkdir(parents=True)
            reports.mkdir(parents=True)
            # EPL has no dedicated league tab; it must still appear in the historical tab.
            pd.DataFrame(
                [{
                    "canonical_game_key": "soccer|EPL|2020-08-01|ARSENAL|CHELSEA",
                    "league": "EPL", "game_date": "2020-08-01",
                    "home_team": "ARSENAL", "away_team": "CHELSEA", "source": "football_data",
                    "outcome_available": True, "game_odds_available": True,
                    "prop_odds_available": False, "closing_available": True,
                }]
            ).to_csv(processed / "historical_game_inventory.csv", index=False)
            html = build_dashboard_html(reports)

        self.assertIn('data-tab="historical-backfill"', html)
        self.assertIn('id="league-historical-backfill"', html)
        self.assertIn("By League and Status", html)
        self.assertIn("soccer|EPL|2020-08-01|ARSENAL|CHELSEA", html)
        # No audit file present -> no quality banner.
        self.assertNotIn("data quality:", html)

    def test_historical_backfill_shows_quality_warning_banner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            reports = Path(tmp) / "reports"
            processed.mkdir(parents=True)
            reports.mkdir(parents=True)
            pd.DataFrame(
                [{
                    "canonical_game_key": "soccer|EPL|2020-08-01|ARSENAL|CHELSEA",
                    "league": "EPL", "game_date": "2020-08-01",
                    "home_team": "ARSENAL", "away_team": "CHELSEA", "source": "football_data",
                    "outcome_available": True, "game_odds_available": True,
                    "prop_odds_available": False, "closing_available": True,
                }]
            ).to_csv(processed / "historical_game_inventory.csv", index=False)
            (reports / "historical_game_inventory_quality_audit.json").write_text(
                '{"verdict": "usable_with_warnings", "critical_issues": [], '
                '"warning_issues": ["44048 rows flag closing but have no game odds"]}',
                encoding="utf-8",
            )
            html = build_dashboard_html(reports)

        self.assertIn("data quality: usable_with_warnings", html)
        self.assertIn("closing but have no game odds", html)
        self.assertIn("historical_game_inventory_quality_audit.md", html)


class TestRecordedGamesPerLeagueFilter(unittest.TestCase):
    K_NBA = "basketball|NBA|2026-06-13|MIA|DEN"
    K_MLB = "baseball|MLB|2026-06-13|NYY|BOS"

    def _write_two_league_fixture(self, root: Path) -> Path:
        processed = root / "processed"
        reports = root / "reports"
        processed.mkdir(parents=True)
        reports.mkdir(parents=True)
        pd.DataFrame(
            [
                {"snapshot_time": "2026-06-13T15:00:00Z", "canonical_game_key": self.K_NBA,
                 "is_closing_snapshot": False, "minutes_to_game_start": 200, "bookmaker": "betmgm", "prop_type": "points"},
                {"snapshot_time": "2026-06-13T15:00:00Z", "canonical_game_key": self.K_MLB,
                 "is_closing_snapshot": False, "minutes_to_game_start": 200, "bookmaker": "betmgm", "prop_type": "hits"},
            ]
        ).to_csv(processed / "player_prop_snapshots_normalized.csv", index=False)
        return reports

    def test_section_filtered_to_one_league(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = self._write_two_league_fixture(Path(tmp))
            frame = _recorded_games_frame(reports)

        nba = _render_recorded_games_section(frame, league="NBA")
        mlb = _render_recorded_games_section(frame, league="MLB")

        # Each league view shows exactly its own game and excludes the other's.
        self.assertIn("Games tracked: 1", nba)
        self.assertIn(self.K_NBA, nba)
        self.assertNotIn(self.K_MLB, nba)

        self.assertIn("Games tracked: 1", mlb)
        self.assertIn(self.K_MLB, mlb)
        self.assertNotIn(self.K_NBA, mlb)

        # The unfiltered view (league=None) still shows both.
        both = _render_recorded_games_section(frame, league=None)
        self.assertIn("Games tracked: 2", both)
        self.assertIn(self.K_NBA, both)
        self.assertIn(self.K_MLB, both)

    def test_league_with_no_recorded_games_shows_named_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = self._write_two_league_fixture(Path(tmp))
            frame = _recorded_games_frame(reports)
        out = _render_recorded_games_section(frame, league="WORLD_CUP")
        self.assertIn("Games tracked: 0", out)
        self.assertIn("No recorded games for WORLD_CUP yet", out)

    def test_live_dashboard_panels_do_not_leak_keys_across_leagues(self) -> None:
        html = build_dashboard_html(PROJECT_ROOT / "data" / "reports")
        nba_panel = _panel_html(html, "nba")
        mlb_panel = _panel_html(html, "mlb")

        self.assertIn("Recorded Games", nba_panel)
        self.assertIn("Recorded Games", mlb_panel)
        # NBA keys appear only under the NBA tab; MLB keys never appear under NBA.
        self.assertIn("basketball|NBA|", nba_panel)
        self.assertNotIn("basketball|NBA|", mlb_panel)
        self.assertNotIn("baseball|MLB|", nba_panel)


if __name__ == "__main__":
    unittest.main()
