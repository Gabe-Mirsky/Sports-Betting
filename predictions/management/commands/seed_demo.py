"""Seed a few example predictions so the site is usable without the pipeline.

Usage::

    python manage.py seed_demo

Safe to re-run: rows are matched on ``fixture_id`` and updated in place.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from django.core.management.base import BaseCommand

from predictions.models import MatchupPrediction

_NOW = datetime.now().replace(minute=0, second=0, microsecond=0)

_SEED = [
    {
        "fixture_id": "demo_soccer_intl_ecuador_curacao",
        "sport": "soccer",
        "league": "international",
        "game_date": _NOW + timedelta(days=1),
        "team_a": "Ecuador",
        "team_b": "Curacao",
        "prob_team_a_win": 0.838,
        "prob_draw": 0.117,
        "prob_team_b_win": 0.045,
        "predicted_outcome": "Ecuador win",
        "confidence_level": "High",
        "confidence_score": 0.72,
        "data_quality": "usable",
        "key_reasons": "Ecuador has the stronger long-term team rating.; Ecuador has better recent form.; Neutral site reduces home advantage.",
        "main_risks": "Curacao can sit deep and frustrate the favourite.",
        "model_version": "matchup_baseline_v1",
    },
    {
        "fixture_id": "demo_soccer_intl_germany_ivory_coast",
        "sport": "soccer",
        "league": "international",
        "game_date": _NOW + timedelta(days=1, hours=3),
        "team_a": "Germany",
        "team_b": "Ivory Coast",
        "prob_team_a_win": 0.615,
        "prob_draw": 0.208,
        "prob_team_b_win": 0.177,
        "predicted_outcome": "Germany win",
        "confidence_level": "High",
        "confidence_score": 0.41,
        "data_quality": "usable",
        "key_reasons": "Germany has the stronger long-term team rating.; Germany has better recent form.",
        "main_risks": "Ivory Coast has more rest before this game.",
        "model_version": "matchup_baseline_v1",
    },
    {
        "fixture_id": "demo_nba_lakers_celtics",
        "sport": "basketball",
        "league": "NBA",
        "game_date": _NOW + timedelta(days=2),
        "team_a": "Los Angeles Lakers",
        "team_b": "Boston Celtics",
        "prob_team_a_win": 0.46,
        "prob_draw": 0.0,
        "prob_team_b_win": 0.54,
        "predicted_outcome": "Boston Celtics win",
        "confidence_level": "Low",
        "confidence_score": 0.08,
        "data_quality": "usable",
        "key_reasons": "Boston has the better recent net rating.; Home court favours Boston.",
        "main_risks": "Back-to-back fatigue could swing a close game.; Star availability is uncertain.",
        "model_version": "matchup_baseline_v1",
    },
]


class Command(BaseCommand):
    help = "Create a small set of example predictions for local demos."

    def handle(self, *args, **options):
        created = updated = 0
        for item in _SEED:
            fixture_id = item.pop("fixture_id")
            item["status"] = MatchupPrediction.STATUS_APPROVED
            _, was_created = MatchupPrediction.objects.update_or_create(
                fixture_id=fixture_id, defaults=item
            )
            created += int(was_created)
            updated += int(not was_created)
        self.stdout.write(
            self.style.SUCCESS(
                f"Seed complete: {created} created, {updated} updated."
            )
        )
