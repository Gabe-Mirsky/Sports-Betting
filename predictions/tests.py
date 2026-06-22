"""Tests for the predictions Django app."""

from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import TestCase
from django.urls import reverse

from .models import MatchupPrediction, Parlay, ParlayLeg
from .services.ai_reasoning import define_prediction_reasoning
from .services.parlay_recommender import recommend_parlays


def _make_prediction(**overrides) -> MatchupPrediction:
    data = {
        "fixture_id": "test_1",
        "sport": "soccer",
        "league": "international",
        "team_a": "Japan",
        "team_b": "Tunisia",
        "prob_team_a_win": 0.65,
        "prob_draw": 0.24,
        "prob_team_b_win": 0.11,
        "predicted_outcome": "Japan win",
        "confidence_level": "Medium",
        "confidence_score": 0.42,
        "data_quality": "usable",
        "key_reasons": "Japan has better recent form.; Japan has more rest.",
        "main_risks": "Friendlies rotate lineups.",
        "status": MatchupPrediction.STATUS_APPROVED,
    }
    data.update(overrides)
    return MatchupPrediction.objects.create(**data)


class ModelTests(TestCase):
    def test_pick_probability_matches_predicted_outcome(self):
        pred = _make_prediction()
        self.assertAlmostEqual(pred.pick_probability, 0.65)
        self.assertEqual(pred.pick_probability_pct, 65.0)

    def test_pick_probability_for_team_b_and_draw(self):
        b = _make_prediction(fixture_id="b", predicted_outcome="Tunisia win")
        self.assertAlmostEqual(b.pick_probability, 0.11)
        d = _make_prediction(fixture_id="d", predicted_outcome="Draw")
        self.assertAlmostEqual(d.pick_probability, 0.24)

    def test_reasons_list_splits_on_semicolons(self):
        pred = _make_prediction()
        self.assertEqual(
            pred.reasons_list,
            ["Japan has better recent form.", "Japan has more rest."],
        )


class ParlayTests(TestCase):
    def test_combined_probability_and_fair_odds(self):
        p1 = _make_prediction(fixture_id="p1")
        p2 = _make_prediction(fixture_id="p2", predicted_outcome="Tunisia win")
        parlay = Parlay.objects.create(name="Test")
        ParlayLeg.objects.create(parlay=parlay, prediction=p1, pick_label="Japan win", pick_probability=0.5)
        ParlayLeg.objects.create(parlay=parlay, prediction=p2, pick_label="Tunisia win", pick_probability=0.5)
        self.assertAlmostEqual(parlay.combined_probability, 0.25)
        self.assertEqual(parlay.combined_probability_pct, 25.0)
        self.assertAlmostEqual(parlay.fair_decimal_odds, 4.0)
        self.assertEqual(parlay.fair_american_odds, "+300")


class AiServiceTests(TestCase):
    def test_fallback_returns_useful_reasoning_without_api_key(self):
        result = define_prediction_reasoning(
            {"team_a": "Japan", "team_b": "Tunisia", "prob_team_a_win": 0.7, "prob_team_b_win": 0.3}
        )
        self.assertIn("key_reasons", result)
        self.assertTrue(result["key_reasons"])
        self.assertEqual(result["provider"], "offline-fallback")
        self.assertTrue(any("Japan" in r for r in result["key_reasons"]))


class ViewTests(TestCase):
    def test_home_and_list_show_approved_only(self):
        _make_prediction(fixture_id="approved")
        _make_prediction(fixture_id="hidden", status=MatchupPrediction.STATUS_HIDDEN, team_a="Hiddenland")
        list_resp = self.client.get(reverse("predictions:list"))
        self.assertEqual(list_resp.status_code, 200)
        self.assertContains(list_resp, "Japan")
        self.assertNotContains(list_resp, "Hiddenland")
        # Simplified column headers are present.
        for header in ["Date", "Sport", "League", "Teams", "Prediction", "Reasoning"]:
            self.assertContains(list_resp, f">{header}</th>")

    def test_detail_page_renders(self):
        pred = _make_prediction()
        resp = self.client.get(pred.get_absolute_url())
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Japan has better recent form.")

    def test_parlay_creator_saves_selected_legs(self):
        p1 = _make_prediction(fixture_id="p1")
        p2 = _make_prediction(fixture_id="p2", team_a="Brazil", team_b="Peru",
                              predicted_outcome="Brazil win")
        resp = self.client.post(
            reverse("predictions:parlay_creator"),
            {"legs": [p1.pk, p2.pk], "name": "My parlay", "notes": ""},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Parlay.objects.count(), 1)
        self.assertEqual(ParlayLeg.objects.count(), 2)

    def test_parlay_creator_rejects_single_leg(self):
        p1 = _make_prediction(fixture_id="p1")
        resp = self.client.post(
            reverse("predictions:parlay_creator"),
            {"legs": [p1.pk], "name": "Too small"},
            follow=True,
        )
        self.assertEqual(Parlay.objects.count(), 0)
        self.assertContains(resp, "at least two")

    def test_add_prediction_flow_creates_pending(self):
        add_resp = self.client.post(
            reverse("predictions:add"),
            {
                "sport": "soccer",
                "league": "international",
                "game_date": "2026-07-01T18:00",
                "team_a": "Spain",
                "team_b": "Italy",
                "prob_team_a_win": "0.6",
                "prob_draw": "0.25",
                "prob_team_b_win": "0.15",
                "predicted_outcome": "",
            },
        )
        self.assertRedirects(add_resp, reverse("predictions:preview"))
        confirm = self.client.post(reverse("predictions:preview"), follow=True)
        self.assertEqual(confirm.status_code, 200)
        created = MatchupPrediction.objects.get(team_a="Spain")
        self.assertEqual(created.status, MatchupPrediction.STATUS_PENDING)
        self.assertEqual(created.predicted_outcome, "Spain win")
        self.assertTrue(created.key_reasons)


class ParlayRecommenderTests(TestCase):
    def test_recommends_parlays_with_reasoning(self):
        _make_prediction(fixture_id="r1", team_a="A1", team_b="B1", prob_team_a_win=0.8,
                         prob_draw=0.1, prob_team_b_win=0.1, predicted_outcome="A1 win",
                         confidence_level="High", league="L1")
        _make_prediction(fixture_id="r2", team_a="A2", team_b="B2", prob_team_a_win=0.75,
                         prob_draw=0.1, prob_team_b_win=0.15, predicted_outcome="A2 win",
                         confidence_level="High", league="L2")
        _make_prediction(fixture_id="r3", team_a="A3", team_b="B3", prob_team_a_win=0.7,
                         prob_draw=0.1, prob_team_b_win=0.2, predicted_outcome="A3 win",
                         confidence_level="Medium", league="L1")
        recs = recommend_parlays(list(MatchupPrediction.objects.all()))
        self.assertTrue(recs)
        top = recs[0]
        self.assertGreaterEqual(len(top.legs), 2)
        self.assertTrue(top.rationale)  # has plain-language reasoning
        # Combined probability is the product of leg probabilities.
        expected = 1.0
        for leg in top.legs:
            expected *= leg.probability
        self.assertAlmostEqual(top.combined_probability, expected, places=6)

    def test_no_recommendations_when_picks_are_weak(self):
        _make_prediction(fixture_id="w1", prob_team_a_win=0.34, prob_draw=0.33,
                         prob_team_b_win=0.33, predicted_outcome="Japan win")
        self.assertEqual(recommend_parlays(list(MatchupPrediction.objects.all())), [])


class HowItWorksViewTests(TestCase):
    def test_page_lists_variables_in_plain_language(self):
        resp = self.client.get(reverse("predictions:how_it_works"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "What the model looks at")
        self.assertContains(resp, "Team A strength score")
        self.assertContains(resp, "How we turn clues into a percent")


class RecommendationsViewTests(TestCase):
    def test_recommendations_page_renders(self):
        _make_prediction(fixture_id="r1", team_a="A1", team_b="B1", prob_team_a_win=0.8,
                         prob_draw=0.1, prob_team_b_win=0.1, predicted_outcome="A1 win",
                         confidence_level="High")
        _make_prediction(fixture_id="r2", team_a="A2", team_b="B2", prob_team_a_win=0.75,
                         prob_draw=0.1, prob_team_b_win=0.15, predicted_outcome="A2 win",
                         confidence_level="High")
        resp = self.client.get(reverse("predictions:recommendations"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Recommended parlays")
        self.assertContains(resp, "Why this parlay")


class ImportCommandTests(TestCase):
    def test_import_predictions_creates_rows(self):
        from django.core.management import call_command

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "preds.csv"
            csv_path.write_text(
                "id,sport,league,game_date,team_a,team_b,prob_team_a_win,prob_draw,prob_team_b_win,"
                "predicted_outcome,confidence_level,confidence_score,data_quality,key_reasons,main_risks\n"
                "fx1,soccer,international,2026-06-20T00:00:00,Ecuador,Curacao,0.84,0.12,0.04,"
                "Ecuador win,High,0.72,usable,Ecuador stronger.,Curacao defends deep.\n",
                encoding="utf-8",
            )
            call_command("import_predictions", path=str(csv_path))

        pred = MatchupPrediction.objects.get(fixture_id="fx1")
        self.assertEqual(pred.team_a, "Ecuador")
        self.assertAlmostEqual(pred.prob_team_a_win, 0.84)
        self.assertEqual(pred.status, MatchupPrediction.STATUS_APPROVED)
        self.assertIsNotNone(pred.game_date)
