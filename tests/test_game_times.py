from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

import data.game_times as game_times  # noqa: E402
from data.game_times import _official_start_time_to_utc, add_game_start_times  # noqa: E402


class TestGameTimes(unittest.TestCase):
    def test_official_start_time_converts_eastern_to_utc(self) -> None:
        self.assertEqual(
            _official_start_time_to_utc("2026-01-01 19:30:00"),
            "2026-01-02T00:30:00+00:00",
        )

    def test_add_game_start_times_merges_official_source_columns(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": "2026-01-01",
                    "home_team_abbr": "NYK",
                    "away_team_abbr": "BOS",
                }
            ]
        )
        starts = pd.DataFrame(
            [
                {
                    "game_date": "2026-01-01",
                    "home_team_abbr": "NYK",
                    "away_team_abbr": "BOS",
                    "game_start_time": "2026-01-02T00:30:00+00:00",
                    "game_time_source": "nba_stats_scoreboard",
                    "nba_game_id": "0022600001",
                }
            ]
        )

        enriched = add_game_start_times(games, starts)

        self.assertEqual(enriched.loc[0, "game_start_time"], "2026-01-02T00:30:00+00:00")
        self.assertEqual(enriched.loc[0, "game_time_source"], "nba_stats_scoreboard")
        self.assertEqual(enriched.loc[0, "nba_game_id"], "0022600001")

    def test_failed_refresh_does_not_overwrite_existing_cache(self) -> None:
        path = PROJECT_ROOT / "data" / "reports" / "_test_game_times.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "game_date,home_team_abbr,away_team_abbr,game_start_time,game_time_source\n"
            "2026-01-01,NYK,BOS,2026-01-02T00:30:00+00:00,existing\n",
            encoding="utf-8",
        )
        failure_path = path.with_name(f"{path.stem}_failures.csv")
        original_fetch = game_times.fetch_game_start_times_for_date
        try:
            game_times.fetch_game_start_times_for_date = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
            output = game_times.download_game_start_times_for_games(
                pd.DataFrame(
                    [
                        {
                            "game_date": "2026-01-01",
                            "home_team_abbr": "NYK",
                            "away_team_abbr": "BOS",
                        }
                    ]
                ),
                output_path=path,
                sleep_seconds=0,
            )
            cached_text = path.read_text(encoding="utf-8")
        finally:
            game_times.fetch_game_start_times_for_date = original_fetch
            path.unlink(missing_ok=True)
            failure_path.unlink(missing_ok=True)

        self.assertTrue(output.empty)
        self.assertIn("existing", cached_text)

    def test_start_time_fallback_labels_espn_source(self) -> None:
        original_official = game_times.fetch_nba_official_scoreboard_date
        original_espn = game_times.fetch_espn_scoreboard_date
        try:
            game_times.fetch_nba_official_scoreboard_date = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
            game_times.fetch_espn_scoreboard_date = lambda *_args, **_kwargs: pd.DataFrame(
                [
                    {
                        "game_date": "2026-01-01",
                        "home_team_abbr": "NYK",
                        "away_team_abbr": "BOS",
                        "game_start_time": "2026-01-02T00:30:00Z",
                        "game_time_source": "espn_scoreboard",
                    }
                ]
            )

            output = game_times.fetch_game_start_times_for_date("2026-01-01")
        finally:
            game_times.fetch_nba_official_scoreboard_date = original_official
            game_times.fetch_espn_scoreboard_date = original_espn

        self.assertEqual(output.loc[0, "game_time_source"], "espn_scoreboard_after_nba_stats_fallback")


if __name__ == "__main__":
    unittest.main()
