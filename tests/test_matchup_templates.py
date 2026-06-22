from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from data.fixtures_loader import (  # noqa: E402
    FIXTURE_COLUMNS,
    load_fixtures,
    normalize_fixtures,
    validate_fixtures,
)
from data.injuries_loader import INJURY_COLUMNS, load_injuries, normalize_injuries  # noqa: E402
from data.match_results_loader import (  # noqa: E402
    MATCH_RESULTS_COLUMNS,
    load_match_results,
    normalize_match_results,
    validate_match_results,
)
from features.matchup_features import build_fixture_features, build_training_features  # noqa: E402
from models.matchup_model import (  # noqa: E402
    PROB_COLUMNS,
    predict_matchup_probabilities,
    train_matchup_model,
)

MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
RESULTS_TEMPLATE = MANUAL_DIR / "match_results_template.csv"
FIXTURES_TEMPLATE = MANUAL_DIR / "fixtures_today_template.csv"
INJURIES_TEMPLATE = MANUAL_DIR / "injuries_template.csv"


class TestTemplatesExist(unittest.TestCase):
    def test_all_templates_present(self) -> None:
        self.assertTrue(RESULTS_TEMPLATE.exists())
        self.assertTrue(FIXTURES_TEMPLATE.exists())
        self.assertTrue(INJURIES_TEMPLATE.exists())


class TestTemplatesLoadable(unittest.TestCase):
    def test_results_template_normalizes(self) -> None:
        normalized = normalize_match_results(load_match_results(RESULTS_TEMPLATE))
        for column in MATCH_RESULTS_COLUMNS:
            self.assertIn(column, normalized.columns)
        self.assertTrue(validate_match_results(normalized)["ok"])
        # Soccer template should contain at least one draw row.
        self.assertGreaterEqual(int(normalized["result_draw"].sum()), 1)

    def test_fixtures_template_normalizes(self) -> None:
        normalized = normalize_fixtures(load_fixtures(FIXTURES_TEMPLATE))
        for column in FIXTURE_COLUMNS:
            self.assertIn(column, normalized.columns)
        self.assertTrue(validate_fixtures(normalized)["ok"])
        # Japan vs Tunisia should be present.
        pair = set(zip(normalized["team_a"], normalized["team_b"]))
        self.assertIn(("Japan", "Tunisia"), pair)

    def test_injuries_template_normalizes(self) -> None:
        normalized = normalize_injuries(load_injuries(INJURIES_TEMPLATE))
        for column in INJURY_COLUMNS:
            self.assertIn(column, normalized.columns)
        statuses = set(normalized["status"].str.lower())
        # Template demonstrates the full status vocabulary.
        self.assertTrue({"out", "doubtful", "questionable", "probable", "available"} <= statuses)


class TestTemplatePredictionFlow(unittest.TestCase):
    def test_templates_flow_to_probabilities_without_odds(self) -> None:
        results = normalize_match_results(load_match_results(RESULTS_TEMPLATE))
        fixtures = normalize_fixtures(load_fixtures(FIXTURES_TEMPLATE))

        training = build_training_features(results)
        bundle = train_matchup_model(training, "soccer")
        fixture_features = build_fixture_features(fixtures, results)
        preds = predict_matchup_probabilities(bundle, fixture_features)

        # Probabilities sum to ~1 for every soccer fixture and include a draw.
        self.assertTrue((preds[PROB_COLUMNS].sum(axis=1).round(6) == 1.0).all())
        self.assertIn("prob_draw", preds.columns)
        self.assertTrue((preds["prob_draw"] > 0).any())

        # The whole flow must not require any odds/market columns.
        odds_like = {"odds", "price", "clv", "implied_prob", "american_odds", "decimal_odds",
                     "closing_line", "vig", "no_vig_prob"}
        self.assertEqual(odds_like & set(c.lower() for c in preds.columns), set())


if __name__ == "__main__":
    unittest.main()
