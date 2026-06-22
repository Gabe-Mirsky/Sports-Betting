from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from models.prediction_explainer import (  # noqa: E402
    assign_confidence_level,
    detect_data_quality_warnings,
)
from quality.matchup_data_quality import assign_prediction_data_quality  # noqa: E402


def _row(**overrides) -> pd.Series:
    base = {
        "team_a": "Japan",
        "team_b": "Tunisia",
        "sport": "soccer",
        "competition_type": "qualifier",
        "min_recent_games": 12,
        "team_a_recent_games": 12,
        "team_b_recent_games": 12,
        "injury_data_present": 1,
        "team_a_injury_stale": False,
        "team_b_injury_stale": False,
        "venue": "Stadium",
        "confidence_score": 0.4,
    }
    base.update(overrides)
    return pd.Series(base)


class TestDataQualityTiers(unittest.TestCase):
    def test_no_history_is_very_weak(self) -> None:
        self.assertEqual(assign_prediction_data_quality(_row(min_recent_games=0)), "very_weak")

    def test_low_history_is_weak(self) -> None:
        self.assertEqual(assign_prediction_data_quality(_row(min_recent_games=3)), "weak")

    def test_full_context_is_strong(self) -> None:
        self.assertEqual(assign_prediction_data_quality(_row()), "strong")

    def test_friendly_without_injuries_downgrades(self) -> None:
        quality = assign_prediction_data_quality(
            _row(competition_type="friendly", injury_data_present=0)
        )
        self.assertIn(quality, {"weak", "usable"})
        self.assertNotEqual(quality, "strong")


class TestDataQualityWarnings(unittest.TestCase):
    def test_missing_injury_data_flagged(self) -> None:
        warnings = detect_data_quality_warnings(_row(injury_data_present=0))
        self.assertTrue(any("No injury data" in w for w in warnings))

    def test_stale_injury_data_flagged(self) -> None:
        warnings = detect_data_quality_warnings(
            _row(injury_data_present=1, team_a_injury_stale=True)
        )
        self.assertTrue(any("older than 48 hours" in w for w in warnings))

    def test_low_recent_games_flagged(self) -> None:
        warnings = detect_data_quality_warnings(_row(team_a_recent_games=2))
        self.assertTrue(any("fewer than 5 recent games" in w for w in warnings))

    def test_not_backtested_flagged(self) -> None:
        warnings = detect_data_quality_warnings(_row(model_backtested=False))
        self.assertTrue(any("not been backtested" in w for w in warnings))


class TestConfidence(unittest.TestCase):
    def test_high_separation_is_high_confidence(self) -> None:
        self.assertEqual(assign_confidence_level(_row(confidence_score=0.5)), "High")

    def test_weak_data_caps_confidence(self) -> None:
        level = assign_confidence_level(_row(confidence_score=0.5, min_recent_games=3))
        self.assertIn(level, {"Low", "Very low"})

    def test_close_game_is_low_confidence(self) -> None:
        self.assertIn(assign_confidence_level(_row(confidence_score=0.01)), {"Low", "Very low"})


if __name__ == "__main__":
    unittest.main()
