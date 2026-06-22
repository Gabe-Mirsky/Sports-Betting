"""Train the matchup model and export its coefficients in a readable form.

Writes ``REPORTS_DIR/model_explanation.json``, which the "How predictions work"
page reads to show the real learned weight of each variable ("points toward
Team A winning"). Re-run after the model or data changes.

Usage::

    python manage.py export_model_explanation
    python manage.py export_model_explanation --results-path data/processed/match_results.csv
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

PROJECT_ROOT = Path(settings.BASE_DIR)
SRC_DIR = PROJECT_ROOT / "src"


def _toward_team_a_weights(clf, feature_names):
    """Return {feature_name: weight toward 'Team A wins'} from a fitted classifier.

    Logistic regression coefficients are on standardized features, so the sign and
    size are directly comparable: positive pushes the result toward Team A.
    """

    import numpy as np

    classes = list(clf.classes_)
    coef = clf.coef_  # shape (n_classes, n_feat) multinomial, or (1, n_feat) binary
    CLASS_TEAM_A = 0
    if coef.shape[0] == 1:
        # Binary: the single row points toward classes_[1] (the non-Team-A class),
        # so flip the sign to express it as "toward Team A".
        weights = -coef[0]
        intercept = -float(clf.intercept_[0])
    else:
        idx = classes.index(CLASS_TEAM_A) if CLASS_TEAM_A in classes else 0
        weights = coef[idx]
        intercept = float(clf.intercept_[idx])
    return {name: float(w) for name, w in zip(feature_names, np.asarray(weights))}, intercept


class Command(BaseCommand):
    help = "Train the matchup model and export readable coefficients to JSON."

    def add_arguments(self, parser):
        parser.add_argument("--results-path", default=str(PROJECT_ROOT / "data" / "processed" / "match_results.csv"))
        parser.add_argument("--injuries-path", default=str(PROJECT_ROOT / "data" / "processed" / "injuries.csv"))
        parser.add_argument("--output-path", default=str(settings.REPORTS_DIR / "model_explanation.json"))

    def handle(self, *args, **options):
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        try:
            from data.match_results_loader import load_match_results, normalize_match_results
            from data.injuries_loader import load_injuries, normalize_injuries
            from features.matchup_features import build_training_features
            from models.matchup_model import train_matchup_model
        except Exception as exc:  # pragma: no cover - import guard
            raise CommandError(f"Could not import the model pipeline from src/: {exc}") from exc

        results_path = Path(options["results_path"])
        if not results_path.exists():
            raise CommandError(f"Match results file not found: {results_path}")

        config: dict = {}
        results = normalize_match_results(load_match_results(str(results_path)), config)
        injuries_path = Path(options["injuries_path"])
        injuries = (
            normalize_injuries(load_injuries(str(injuries_path)), config)
            if injuries_path.exists()
            else None
        )

        sports_out: dict = {}
        for sport in sorted(results["sport"].dropna().unique()):
            sport_results = results[results["sport"] == sport]
            if len(sport_results) < 30:
                self.stdout.write(f"Skipping {sport}: only {len(sport_results)} games.")
                continue
            try:
                training = build_training_features(sport_results, injuries, config)
                bundle = train_matchup_model(training, sport, config)
            except Exception as exc:
                self.stdout.write(self.style.WARNING(f"Skipping {sport}: training failed ({exc})."))
                continue

            pipeline = bundle["model"]
            pre = pipeline.named_steps["pre"]
            clf = pipeline.named_steps["clf"]
            if not hasattr(clf, "coef_"):
                self.stdout.write(self.style.WARNING(f"Skipping {sport}: model has no coefficients (not logistic)."))
                continue

            raw_names = list(pre.get_feature_names_out())
            # Strip the ColumnTransformer prefixes ("num__"/"cat__").
            clean_names = [n.split("__", 1)[-1] for n in raw_names]
            weights, intercept = _toward_team_a_weights(clf, clean_names)

            ordered = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
            sports_out[sport] = {
                "model_type": bundle.get("model_type"),
                "model_version": bundle.get("model_version"),
                "draws_enabled": bundle.get("draws_enabled"),
                "n_train": bundle.get("n_train"),
                "intercept_toward_team_a": round(intercept, 4),
                "weights": {name: round(w, 4) for name, w in weights.items()},
                "top_factors": [{"feature": n, "weight": round(w, 4)} for n, w in ordered[:8]],
            }
            self.stdout.write(f"Exported {sport}: {len(weights)} weights from {bundle.get('n_train')} games.")

        if not sports_out:
            raise CommandError("No sports could be trained; nothing exported.")

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "explanation": (
                "Each number is the learned weight of a clue, on a standardized scale, "
                "expressed as points toward Team A winning. Positive points push the "
                "prediction toward Team A; negative points push it toward the other team."
            ),
            "sports": sports_out,
        }
        out_path = Path(options["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Wrote {out_path}"))
