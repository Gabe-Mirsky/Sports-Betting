"""Import matchup predictions from the analysis pipeline's report CSV.

Usage::

    python manage.py import_predictions
    python manage.py import_predictions --path data/reports/matchup_predictions_today.csv
    python manage.py import_predictions --status pending   # require manual approval

Existing rows (matched on ``fixture_id``) are updated in place so re-running the
pipeline + this command keeps the site current without creating duplicates.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from predictions.models import MatchupPrediction

# CSV column -> model field. Columns absent from the CSV are simply skipped.
_FIELD_MAP = {
    "sport": "sport",
    "league": "league",
    "team_a": "team_a",
    "team_b": "team_b",
    "predicted_outcome": "predicted_outcome",
    "confidence_level": "confidence_level",
    "data_quality": "data_quality",
    "key_reasons": "key_reasons",
    "main_risks": "main_risks",
    "data_quality_warnings": "data_quality_warnings",
    "model_version": "model_version",
}
_FLOAT_FIELDS = {
    "prob_team_a_win",
    "prob_draw",
    "prob_team_b_win",
    "confidence_score",
}
_DATE_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")


def _parse_float(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(value: str):
    value = (value or "").strip()
    if not value:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class Command(BaseCommand):
    help = "Import matchup predictions from the pipeline report CSV into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default=None,
            help="CSV path (default: REPORTS_DIR/matchup_predictions_today.csv).",
        )
        parser.add_argument(
            "--status",
            default=MatchupPrediction.STATUS_APPROVED,
            choices=[c[0] for c in MatchupPrediction.STATUS_CHOICES],
            help="Status to assign to NEW rows (default: approved).",
        )

    def handle(self, *args, **options):
        path = Path(options["path"]) if options["path"] else settings.REPORTS_DIR / "matchup_predictions_today.csv"
        if not path.exists():
            raise CommandError(f"Predictions CSV not found: {path}")

        new_status = options["status"]
        created = updated = skipped = 0

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                fixture_id = (row.get("id") or row.get("fixture_id") or "").strip()
                team_a = (row.get("team_a") or "").strip()
                team_b = (row.get("team_b") or "").strip()
                if not fixture_id or not team_a or not team_b:
                    skipped += 1
                    continue

                defaults = {model_field: (row.get(csv_col) or "").strip()
                            for csv_col, model_field in _FIELD_MAP.items()}
                for field in _FLOAT_FIELDS:
                    defaults[field] = _parse_float(row.get(field, ""))
                defaults["game_date"] = _parse_date(row.get("game_date", ""))

                obj, was_created = MatchupPrediction.objects.update_or_create(
                    fixture_id=fixture_id, defaults=defaults
                )
                if was_created:
                    obj.status = new_status
                    obj.save(update_fields=["status"])
                    created += 1
                else:
                    updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Imported predictions from {path.name}: "
                f"{created} created, {updated} updated, {skipped} skipped."
            )
        )
