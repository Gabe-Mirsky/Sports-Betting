from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from features.team_strength import (  # noqa: E402
    calculate_elo_ratings,
    expected_score,
    get_latest_team_strength,
)

# No home advantage and no MOV so we can reason about exact rating changes.
_FLAT = {"home_advantage": 0.0, "use_margin_of_victory": False, "k_factor": 20.0}


def _two_game_frame(second_winner: str) -> pd.DataFrame:
    """Game 1: A beats B (A becomes the favorite). Game 2: chosen winner."""

    if second_winner == "A":
        g2 = {"team_a_score": 1, "team_b_score": 0, "result_team_a_win": 1, "result_team_b_win": 0}
    else:
        g2 = {"team_a_score": 0, "team_b_score": 1, "result_team_a_win": 0, "result_team_b_win": 1}
    base = {
        "sport": "soccer",
        "team_a": "A",
        "team_b": "B",
        "team_a_home_flag": 0,
        "team_b_home_flag": 0,
        "neutral_site": 1,
        "result_draw": 0,
    }
    return pd.DataFrame(
        [
            {"game_id": "g1", "game_date": "2026-01-01", "team_a_score": 1, "team_b_score": 0,
             "result_team_a_win": 1, "result_draw": 0, "result_team_b_win": 0, **base},
            {"game_id": "g2", "game_date": "2026-01-02", **base, **g2},
        ]
    )


class TestElo(unittest.TestCase):
    def test_expected_score_favors_higher_rating(self) -> None:
        self.assertGreater(expected_score(1700, 1500), 0.5)
        self.assertLess(expected_score(1500, 1700), 0.5)

    def test_favorite_gains_fewer_than_underdog_on_upset(self) -> None:
        fav = calculate_elo_ratings(_two_game_frame("A"), _FLAT)
        ups = calculate_elo_ratings(_two_game_frame("B"), _FLAT)

        fav_row = fav[fav["game_id"] == "g2"].iloc[0]
        favorite_gain = fav_row["team_a_elo_post"] - fav_row["team_a_elo_pre"]

        ups_row = ups[ups["game_id"] == "g2"].iloc[0]
        # In the upset, B (the underdog) wins; measure B's gain.
        underdog_gain = ups_row["team_b_elo_post"] - ups_row["team_b_elo_pre"]

        self.assertLess(favorite_gain, 10.0)  # expected win -> small gain
        self.assertGreater(underdog_gain, 10.0)  # upset -> large gain

    def test_draw_leaves_even_teams_unchanged(self) -> None:
        frame = pd.DataFrame(
            [{"game_id": "d1", "game_date": "2026-01-01", "sport": "soccer",
              "team_a": "A", "team_b": "B", "team_a_score": 1, "team_b_score": 1,
              "team_a_home_flag": 0, "team_b_home_flag": 0, "neutral_site": 1,
              "result_team_a_win": 0, "result_draw": 1, "result_team_b_win": 0}]
        )
        out = calculate_elo_ratings(frame, _FLAT).iloc[0]
        self.assertAlmostEqual(out["team_a_elo_post"], out["team_a_elo_pre"], places=6)
        self.assertAlmostEqual(out["team_b_elo_post"], out["team_b_elo_pre"], places=6)

    def test_neutral_site_has_no_home_advantage(self) -> None:
        config = {"home_advantage": 100.0, "use_margin_of_victory": False}
        frame = pd.DataFrame(
            [
                {"game_id": "home", "game_date": "2026-01-01", "sport": "soccer",
                 "team_a": "A", "team_b": "B", "team_a_score": 1, "team_b_score": 0,
                 "team_a_home_flag": 1, "team_b_home_flag": 0, "neutral_site": 0,
                 "result_team_a_win": 1, "result_draw": 0, "result_team_b_win": 0},
                {"game_id": "neutral", "game_date": "2026-01-01", "sport": "basketball",
                 "team_a": "C", "team_b": "D", "team_a_score": 1, "team_b_score": 0,
                 "team_a_home_flag": 0, "team_b_home_flag": 0, "neutral_site": 1,
                 "result_team_a_win": 1, "result_draw": 0, "result_team_b_win": 0},
            ]
        )
        out = calculate_elo_ratings(frame, config)
        home_expected = out[out["game_id"] == "home"]["elo_expected_a"].iloc[0]
        neutral_expected = out[out["game_id"] == "neutral"]["elo_expected_a"].iloc[0]
        self.assertGreater(home_expected, 0.5)  # home boost applied
        self.assertAlmostEqual(neutral_expected, 0.5, places=6)  # no boost at neutral

    def test_latest_strength_one_row_per_team(self) -> None:
        out = calculate_elo_ratings(_two_game_frame("A"), _FLAT)
        latest = get_latest_team_strength(out)
        self.assertEqual(set(latest["team"]), {"A", "B"})
        self.assertEqual(len(latest), 2)


if __name__ == "__main__":
    unittest.main()
