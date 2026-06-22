"""AI reasoning service for matchup predictions.

Reads ``AI_PROVIDER``, ``AI_API_KEY`` and ``AI_MODEL`` from the environment (via
Django settings). When no API key is configured, it returns a useful,
deterministic set of reasons/risks/narrative derived from the matchup context so
the site keeps working fully offline.

Public entry point: :func:`define_prediction_reasoning`.
"""

from __future__ import annotations

import json
from typing import Any

from django.conf import settings


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _favourite(context: dict[str, Any]) -> tuple[str, str, float | None]:
    """Return (favourite_team, underdog_team, favourite_prob) from probabilities."""

    team_a = context.get("team_a", "Team A")
    team_b = context.get("team_b", "Team B")
    prob_a = _safe_float(context.get("prob_team_a_win"))
    prob_b = _safe_float(context.get("prob_team_b_win"))
    if prob_a is None and prob_b is None:
        return team_a, team_b, None
    if (prob_b or 0) > (prob_a or 0):
        return team_b, team_a, prob_b
    return team_a, team_b, prob_a


def _fallback(context: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, useful reasoning when no AI provider is configured."""

    favourite, underdog, fav_prob = _favourite(context)
    sport = context.get("sport", "the match")
    league = context.get("league", "")
    prob_draw = _safe_float(context.get("prob_draw"))

    reasons = [
        f"{favourite} carries the stronger model win probability in this matchup.",
        f"{favourite}'s recent form and team strength rate above {underdog}'s.",
    ]
    if fav_prob is not None:
        reasons.append(
            f"The model gives {favourite} about {round(fav_prob * 100)}% to win outright."
        )
    if prob_draw and prob_draw > 0.2:
        reasons.append("A draw is a live outcome, so the edge is not one-sided.")

    risks = [
        f"{underdog} can disrupt the favourite if key players are rested or rotated.",
        "Availability and lineup news close to game time can move these probabilities.",
    ]
    if league:
        risks.append(f"{league} scheduling and travel can compress {sport} form signals.")

    narrative = (
        f"The model leans {favourite} over {underdog}. "
        + " ".join(reasons)
        + " Treat this as a probability estimate, not a betting recommendation."
    )

    return {
        "key_reasons": reasons,
        "main_risks": risks,
        "narrative": narrative,
        "provider": "offline-fallback",
        "model": "deterministic",
    }


def _call_anthropic(context: dict[str, Any], api_key: str, model: str) -> dict[str, Any]:
    """Best-effort Anthropic call; raises on any problem so callers can fall back."""

    import anthropic  # imported lazily; optional dependency

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are a sports analyst. Given this matchup context as JSON, return a JSON "
        "object with keys 'key_reasons' (list of short strings), 'main_risks' (list of "
        "short strings) and 'narrative' (one short paragraph). Be concise and never "
        "present this as a betting recommendation.\n\n"
        + json.dumps(context, default=str)
    )
    message = client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    data = json.loads(text)
    data.setdefault("provider", "anthropic")
    data.setdefault("model", model)
    return data


def _call_openai(context: dict[str, Any], api_key: str, model: str) -> dict[str, Any]:
    """Best-effort OpenAI call; raises on any problem so callers can fall back."""

    from openai import OpenAI  # imported lazily; optional dependency

    client = OpenAI(api_key=api_key)
    prompt = (
        "Return a JSON object with keys 'key_reasons' (list of short strings), "
        "'main_risks' (list of short strings) and 'narrative' (one short paragraph) "
        "for this sports matchup. Never present it as a betting recommendation.\n\n"
        + json.dumps(context, default=str)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    data.setdefault("provider", "openai")
    data.setdefault("model", model)
    return data


def define_prediction_reasoning(context: dict[str, Any]) -> dict[str, Any]:
    """Return reasoning for a matchup.

    ``context`` should contain at least team_a/team_b and the win probabilities.
    Always returns a dict with ``key_reasons`` (list), ``main_risks`` (list),
    ``narrative`` (str), ``provider`` (str) and ``model`` (str). Falls back to a
    deterministic local result if no key is set or the provider call fails.
    """

    api_key = getattr(settings, "AI_API_KEY", "") or ""
    provider = (getattr(settings, "AI_PROVIDER", "") or "anthropic").lower()
    model = getattr(settings, "AI_MODEL", "") or ""

    if not api_key:
        return _fallback(context)

    try:
        if provider == "openai":
            return _call_openai(context, api_key, model or "gpt-4o-mini")
        return _call_anthropic(context, api_key, model or "claude-opus-4-8")
    except Exception:
        # Any SDK/network/parse failure degrades gracefully to the local result.
        result = _fallback(context)
        result["provider"] = f"{provider}-unavailable"
        return result
