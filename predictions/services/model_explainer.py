"""Read the exported model coefficients (if present) for the explainer page."""

from __future__ import annotations

import json

from django.conf import settings


def get_model_explanation() -> dict | None:
    """Return the parsed model_explanation.json, or None if it hasn't been exported."""

    path = settings.REPORTS_DIR / "model_explanation.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
