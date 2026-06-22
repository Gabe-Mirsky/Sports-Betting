from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.fixtures_loader import normalize_fixtures  # noqa: E402
from data.match_results_loader import normalize_match_results  # noqa: E402
from features.matchup_features import build_fixture_features, build_training_features  # noqa: E402
from models.matchup_model import (  # noqa: E402
    PROB_COLUMNS,
    normalize_probability_columns,
    predict_matchup_probabilities,
    train_matchup_model,
)


def _synthetic_results(sport: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if sport == "soccer":
        strength = {"A": 2.6, "B": 2.0, "C": 1.4, "D": 1.0}
        base = 0
    else:
        strength = {"A": 12, "B": 6, "C": 2, "D": 0}
        base = 100
    rows = []
    gid = 0
    start = pd.Timestamp("2024-01-01")
    for rnd in range(16):
        for a, b in itertools.permutations(strength, 2):
            gid += 1
            la = base + rng.poisson(strength[a] + 0.3)
            lb = base + rng.poisson(strength[b])
            if sport != "soccer" and la == lb:
                la += 1
            rows.append(
                {
                    "game_id": f"g{gid}",
                    "date": (start + pd.Timedelta(days=gid // 2)).date().isoformat(),
                    "home_team": a,
                    "away_team": b,
                    "home_score": int(la),
                    "away_score": int(lb),
                    "sport": sport,
                    "league": "lg",
                    "competition_type": "league",
                }
            )
    return normalize_match_results(pd.DataFrame(rows), {"default_sport": sport})


class TestProbabilityOutputs(unittest.TestCase):
    def test_normalize_probability_columns_sum_to_one(self) -> None:
        df = pd.DataFrame({"a": [2.0, 0.0], "b": [2.0, 0.0], "c": [0.0, 0.0]})
        out = normalize_probability_columns(df, ["a", "b", "c"])
        self.assertTrue(np.allclose(out[["a", "b", "c"]].sum(axis=1), 1.0))
        # All-zero row falls back to uniform.
        self.assertAlmostEqual(out.loc[1, "a"], 1 / 3)

    def test_three_outcome_probs_sum_to_one(self) -> None:
        results = _synthetic_results("soccer")
        bundle = train_matchup_model(build_training_features(results), "soccer")
        self.assertTrue(bundle["draws_enabled"])
        fixtures = normalize_fixtures(
            pd.DataFrame([{"date": "2024-09-01", "team_a": "A", "team_b": "D",
                           "sport": "soccer", "league": "lg", "competition_type": "league"}])
        )
        preds = predict_matchup_probabilities(bundle, build_fixture_features(fixtures, results))
        self.assertAlmostEqual(float(preds[PROB_COLUMNS].sum(axis=1).iloc[0]), 1.0, places=6)

    def test_two_outcome_probs_sum_to_one_and_draw_zero(self) -> None:
        results = _synthetic_results("basketball")
        bundle = train_matchup_model(build_training_features(results), "basketball")
        self.assertFalse(bundle["draws_enabled"])
        fixtures = normalize_fixtures(
            pd.DataFrame([{"date": "2024-09-01", "team_a": "A", "team_b": "D",
                           "sport": "basketball", "league": "lg", "competition_type": "league"}]),
            {"default_sport": "basketball"},
        )
        preds = predict_matchup_probabilities(bundle, build_fixture_features(fixtures, results))
        self.assertAlmostEqual(float(preds["prob_draw"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(preds[PROB_COLUMNS].sum(axis=1).iloc[0]), 1.0, places=6)

    def test_stronger_team_is_favored(self) -> None:
        results = _synthetic_results("soccer")
        bundle = train_matchup_model(build_training_features(results), "soccer")
        fixtures = normalize_fixtures(
            pd.DataFrame([{"date": "2024-09-01", "team_a": "A", "team_b": "D",
                           "sport": "soccer", "league": "lg", "competition_type": "league"}])
        )
        preds = predict_matchup_probabilities(bundle, build_fixture_features(fixtures, results))
        row = preds.iloc[0]
        self.assertGreater(float(row["prob_team_a_win"]), float(row["prob_team_b_win"]))


if __name__ == "__main__":
    unittest.main()
