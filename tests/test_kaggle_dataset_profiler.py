from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import kaggle_dataset_profiler as profiler  # noqa: E402
from data.kaggle_dataset_profiler import (  # noqa: E402
    build_file_inventory,
    classify_dataset,
    column_signals,
    profile_dataset,
    profile_file,
    resolve_dataset_path,
)


def _write_csv(folder: Path, name: str, frame: pd.DataFrame) -> Path:
    path = folder / name
    frame.to_csv(path, index=False)
    return path


class TestColumnSignals(unittest.TestCase):
    def test_turnovers_not_flagged_as_over_price(self) -> None:
        self.assertNotIn("over_price", column_signals("turnovers"))
        self.assertIn("over_price", column_signals("over_odds"))

    def test_player_and_prop_signals(self) -> None:
        self.assertIn("player_name", column_signals("player_name"))
        self.assertIn("prop_type", column_signals("prop_type"))
        self.assertIn("prop_line", column_signals("line"))


class TestPlayerPropDetection(unittest.TestCase):
    def test_player_prop_dataset_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "player_name": ["LeBron James", "Stephen Curry"],
                    "team": ["LAL", "GSW"],
                    "prop_type": ["points", "points"],
                    "line": [25.5, 27.5],
                    "over_odds": [-110, -115],
                    "under_odds": [-110, -105],
                    "game_date": ["2025-01-01", "2025-01-01"],
                    "result": [30, 22],
                }
            )
            _write_csv(folder, "props.csv", frame)
            result = profile_file(folder / "props.csv", folder, slug="some/nba-props")
            self.assertTrue(result.looks_like_player_props)
            self.assertFalse(result.looks_like_game_odds)
            self.assertEqual(result.detected_market_type, "player_props")
            self.assertTrue(result.can_be_used_for_settlement)

    def test_odds_without_player_fields_is_not_props(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "home_team": ["LAL"],
                    "away_team": ["BOS"],
                    "moneyline_home": [-150],
                    "spread": [-3.5],
                    "total": [220.5],
                }
            )
            _write_csv(folder, "odds.csv", frame)
            result = profile_file(folder / "odds.csv", folder)
            self.assertFalse(result.looks_like_player_props)


class TestGameOddsDetection(unittest.TestCase):
    def test_game_odds_dataset_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "game_date": ["2025-01-01", "2025-01-02"],
                    "home_team": ["LAL", "BOS"],
                    "away_team": ["BOS", "NYK"],
                    "moneyline_home": [-150, 120],
                    "moneyline_away": [130, -140],
                    "spread": [-3.5, 2.5],
                    "total": [220.5, 210.0],
                    "home_score": [110, 101],
                    "away_score": [104, 99],
                }
            )
            _write_csv(folder, "nba_odds.csv", frame)
            result = profile_dataset(str(folder))
            self.assertEqual(result.classification, "game_odds_ready")
            self.assertEqual(result.recommendation, "use_now")
            self.assertEqual(result.files[0].detected_sport, "NBA")


class TestStatsOnlyDetection(unittest.TestCase):
    def test_stats_only_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "player_name": ["Nikola Jokic"],
                    "team": ["DEN"],
                    "pts": [28],
                    "reb": [12],
                    "ast": [9],
                    "min": [35],
                }
            )
            _write_csv(folder, "boxscores.csv", frame)
            result = profile_dataset(str(folder))
            self.assertEqual(result.classification, "stats_only")
            self.assertEqual(result.recommendation, "use_later")
            self.assertFalse(result.files[0].looks_like_player_props)


class TestTimestampOpenCloseDetection(unittest.TestCase):
    def test_open_close_enables_clv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "home_team": ["LAL"],
                    "away_team": ["BOS"],
                    "open_moneyline": [-140],
                    "close_moneyline": [-160],
                }
            )
            _write_csv(folder, "lines.csv", frame)
            result = profile_file(folder / "lines.csv", folder)
            self.assertTrue(result.has_opening_odds)
            self.assertTrue(result.has_closing_odds)
            self.assertTrue(result.can_be_used_for_clv)

    def test_two_timestamps_enable_clv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "team": ["LAL"],
                    "moneyline": [-150],
                    "captured_at": ["2025-01-01T10:00:00Z"],
                    "updated_time": ["2025-01-01T19:00:00Z"],
                }
            )
            _write_csv(folder, "snaps.csv", frame)
            result = profile_file(folder / "snaps.csv", folder)
            self.assertGreaterEqual(result.timestamp_column_count, 2)
            self.assertTrue(result.can_be_used_for_clv)


class TestMissingResultWarning(unittest.TestCase):
    def test_odds_without_results_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame(
                {
                    "home_team": ["LAL", "BOS"],
                    "away_team": ["BOS", "NYK"],
                    "moneyline_home": [-150, 120],
                    "spread": [-3.5, 2.5],
                    "total": [220.5, 210.0],
                }
            )
            _write_csv(folder, "odds_only.csv", frame)
            result = profile_dataset(str(folder))
            self.assertEqual(result.classification, "odds_without_results")
            self.assertIn("no_results_for_grading", result.flags)
            self.assertEqual(result.recommendation, "manual_review")


class TestClassifyHelpers(unittest.TestCase):
    def test_empty_files_unusable(self) -> None:
        self.assertEqual(classify_dataset([]), "unusable_or_unknown")


class TestSlugHandling(unittest.TestCase):
    def test_slug_not_downloaded_by_default(self) -> None:
        path, source, detail = resolve_dataset_path("owner/some-dataset", download=False)
        self.assertEqual(path, "")
        self.assertEqual(source, "kaggle_slug")
        self.assertEqual(detail, "not_downloaded")

    def test_slug_profile_status_without_download(self) -> None:
        result = profile_dataset("ehallmar/nba-historical-stats-and-betting-data", download=False)
        self.assertEqual(result.status, "not_downloaded")
        self.assertEqual(result.recommendation, "manual_review")
        self.assertIn("NBA", result.detected_sports)

    def test_slug_handling_with_mocked_kagglehub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_csv(
                folder,
                "wnba.csv",
                pd.DataFrame(
                    {
                        "home_team": ["LAS"],
                        "away_team": ["NYL"],
                        "moneyline_home": [-120],
                        "spread": [-2.5],
                        "total": [160.5],
                        "winner": ["LAS"],
                    }
                ),
            )
            fake_module = types.ModuleType("kagglehub")
            fake_module.dataset_download = lambda slug: str(folder)  # type: ignore[attr-defined]
            sys.modules["kagglehub"] = fake_module
            try:
                path, source, detail = resolve_dataset_path("zachht/wnba-odds-history", download=True)
                self.assertEqual(path, str(folder))
                self.assertEqual(detail, "downloaded")
                result = profile_dataset("zachht/wnba-odds-history", download=True)
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.classification, "game_odds_ready")
                self.assertEqual(result.files[0].detected_sport, "WNBA")
            finally:
                del sys.modules["kagglehub"]


class TestReports(unittest.TestCase):
    def test_file_inventory_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _write_csv(folder, "a.csv", pd.DataFrame({"team": ["LAL"], "moneyline": [-150], "total": [220], "result": ["W"]}))
            profile = profile_dataset(str(folder))
            inventory = build_file_inventory([profile])
            self.assertIn("looks_like_player_props", inventory.columns)
            self.assertIn("can_be_used_for_clv", inventory.columns)
            self.assertEqual(len(inventory), 1)


if __name__ == "__main__":
    unittest.main()
