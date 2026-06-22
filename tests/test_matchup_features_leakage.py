from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.match_results_loader import normalize_match_results  # noqa: E402
from features.matchup_features import NUMERIC_FEATURES, build_training_features  # noqa: E402
from features.team_strength import add_pre_game_elo_features  # noqa: E402


def _results() -> pd.DataFrame:
    # Team X plays 4 games with deterministic outcomes: W, W, L, W.
    raw = pd.DataFrame(
        [
            {"date": "2026-01-01", "home_team": "X", "away_team": "A", "home_score": 3, "away_score": 0},
            {"date": "2026-01-08", "home_team": "X", "away_team": "B", "home_score": 2, "away_score": 1},
            {"date": "2026-01-15", "home_team": "X", "away_team": "C", "home_score": 0, "away_score": 2},
            {"date": "2026-01-22", "home_team": "X", "away_team": "D", "home_score": 1, "away_score": 0},
        ]
    )
    return normalize_match_results(raw, {"default_sport": "soccer"})


class TestFeatureLeakage(unittest.TestCase):
    def test_pre_game_elo_differs_from_post(self) -> None:
        out = add_pre_game_elo_features(_results())
        decisive = out[out["result_draw"] == 0].iloc[0]
        self.assertNotAlmostEqual(decisive["team_a_elo_pre"], decisive["team_a_elo_post"])

    def test_first_game_uses_no_prior_history(self) -> None:
        feats = build_training_features(_results()).sort_values("game_date").reset_index(drop=True)
        first = feats.iloc[0]
        # X's very first game must not know its own (or any) result yet.
        self.assertEqual(int(first["team_a_recent_games"]), 0)
        self.assertAlmostEqual(first["team_a_recent_win_rate_5"], 0.5)

    def test_rolling_form_uses_only_previous_games(self) -> None:
        feats = build_training_features(_results()).sort_values("game_date").reset_index(drop=True)
        # X's results so far: W, W, L, W. Going into game 3 (index 2), X has W,W
        # => win rate 1.0 over prior games (not influenced by game 3's loss).
        third = feats.iloc[2]
        self.assertEqual(third["team_a"], "X")
        self.assertAlmostEqual(third["team_a_recent_win_rate_5"], 1.0)
        # Going into game 4, prior results W,W,L => win rate 2/3.
        fourth = feats.iloc[3]
        self.assertAlmostEqual(fourth["team_a_recent_win_rate_5"], 2.0 / 3.0)

    def test_no_nans_in_numeric_features(self) -> None:
        feats = build_training_features(_results())
        self.assertEqual(int(feats[NUMERIC_FEATURES].isna().sum().sum()), 0)


if __name__ == "__main__":
    unittest.main()
