"""Read the exported model coefficients (if present) for the explainer page."""

from __future__ import annotations

import json

from django.conf import settings


def get_model_explanation() -> dict | None:
    """Return the parsed model_explanation.json, or None if it hasn't been exported.

    Looks at the live report dir first, then the committed seed copy (so the page
    shows coefficients on hosts where data/reports/ isn't in the repo).
    """

    from pathlib import Path

    candidates = [
        settings.REPORTS_DIR / "model_explanation.json",
        Path(settings.BASE_DIR) / "predictions" / "seed_data" / "model_explanation.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    return None
