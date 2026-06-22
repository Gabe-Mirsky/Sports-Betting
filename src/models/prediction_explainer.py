"""Turn a matchup prediction row into plain-English reasons, risks, warnings.

Everything here is deliberately conservative in tone: we describe *model
probabilities*, *confidence*, and *uncertainty* – never "locks", "guarantees",
or betting language. The functions read whatever feature/context columns are
present on the row and degrade gracefully when some are missing.
"""

from __future__ import annotations

import pandas as pd

from quality.matchup_data_quality import assign_prediction_data_quality

# Confidence ladder, strongest first.
CONFIDENCE_LEVELS = ["High", "Medium", "Low", "Very low"]

# Feature thresholds for when a difference is worth mentioning.
_ELO_GAP = 25.0
_WIN_RATE_GAP = 0.15
_SCORE_GAP = 0.5
_REST_GAP = 2.0


def _get(row, key, default=None):
    try:
        value = row.get(key, default)
    except AttributeError:
        value = default
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def _team_a(row) -> str:
    return str(_get(row, "team_a", "Team A"))


def _team_b(row) -> str:
    return str(_get(row, "team_b", "Team B"))


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _side_availability_present(row, side: str) -> bool:
    key = f"{side}_availability_present"
    if key in getattr(row, "index", []):
        return _boolish(_get(row, key, 0))
    return _boolish(_get(row, "injury_data_present", 0))


def _side_availability_manual(row, side: str) -> bool:
    source = str(_get(row, f"{side}_availability_source", "")).lower()
    return _boolish(_get(row, f"{side}_availability_manual", False)) or "manual" in source


def _has_side_availability_fields(row) -> bool:
    index = getattr(row, "index", [])
    return "team_a_availability_present" in index or "team_b_availability_present" in index


def _availability_context(row) -> list[str]:
    if not _has_side_availability_fields(row):
        if not _boolish(_get(row, "injury_data_present", 0)):
            return ["No injury data available; no confirmed lineup."]
        if _boolish(_get(row, "team_a_injury_stale", False)) or _boolish(_get(row, "team_b_injury_stale", False)):
            return ["Injury data is older than 48 hours."]
        return []

    context: list[str] = []
    for side, team in (("team_a", _team_a(row)), ("team_b", _team_b(row))):
        if not _side_availability_present(row, side):
            context.append(f"No availability data for {team}.")
            continue
        if _boolish(_get(row, f"{side}_injury_stale", False)):
            context.append(f"Availability data for {team} is older than 48 hours.")
        if _side_availability_manual(row, side):
            context.append(f"Injury impact for {team} is based on manual availability data.")
    return context


def _cap(level: str, ceiling: str) -> str:
    """Return the weaker of ``level`` and ``ceiling`` on the confidence ladder."""

    return CONFIDENCE_LEVELS[max(CONFIDENCE_LEVELS.index(level), CONFIDENCE_LEVELS.index(ceiling))]


def _downgrade(level: str, steps: int = 1) -> str:
    idx = min(CONFIDENCE_LEVELS.index(level) + steps, len(CONFIDENCE_LEVELS) - 1)
    return CONFIDENCE_LEVELS[idx]


def assign_confidence_level(row: pd.Series) -> str:
    """Map probability separation + data quality onto a confidence label.

    Confidence reflects *model* confidence and data quality, not bet quality.
    """

    margin = float(_get(row, "confidence_score", 0.0) or 0.0)
    if margin >= 0.30:
        level = "High"
    elif margin >= 0.15:
        level = "Medium"
    elif margin >= 0.06:
        level = "Low"
    else:
        level = "Very low"

    quality = str(_get(row, "data_quality") or assign_prediction_data_quality(row))
    if quality == "very_weak":
        level = _cap(level, "Very low")
    elif quality == "weak":
        level = _cap(level, "Low")

    competition = str(_get(row, "competition_type", "")).lower()
    if "friendly" in competition or "exhibition" in competition:
        level = _downgrade(level, 1)

    injury_stale = bool(_get(row, "team_a_injury_stale", False)) or bool(
        _get(row, "team_b_injury_stale", False)
    )
    if injury_stale and level == "High":
        level = _downgrade(level, 1)

    return level


def _reasons(row) -> list[str]:
    reasons: list[str] = []
    a, b = _team_a(row), _team_b(row)

    elo_diff = float(_get(row, "elo_diff", 0.0) or 0.0)
    if abs(elo_diff) >= _ELO_GAP:
        stronger = a if elo_diff > 0 else b
        reasons.append(f"{stronger} has the stronger long-term team rating.")

    wr_diff = float(_get(row, "recent_win_rate_diff_5", 0.0) or 0.0)
    if abs(wr_diff) >= _WIN_RATE_GAP:
        better = a if wr_diff > 0 else b
        reasons.append(f"{better} has better recent form.")

    score_diff = float(_get(row, "recent_score_diff", 0.0) or 0.0)
    if abs(score_diff) >= _SCORE_GAP:
        better = a if score_diff > 0 else b
        reasons.append(f"{better} has the better recent scoring margin.")

    a_out = int(_get(row, "team_a_key_players_out", 0) or 0)
    b_out = int(_get(row, "team_b_key_players_out", 0) or 0)
    if a_out and a_out > b_out:
        reasons.append(f"{a} has {a_out} key player(s) listed out.")
    elif b_out and b_out > a_out:
        reasons.append(f"{b} has {b_out} key player(s) listed out.")

    rest_diff = float(_get(row, "rest_diff", 0.0) or 0.0)
    if abs(rest_diff) >= _REST_GAP:
        rested = a if rest_diff > 0 else b
        reasons.append(f"{rested} has more rest before this game.")

    neutral = int(_get(row, "neutral_site", 0) or 0)
    if neutral:
        reasons.append("Neutral site reduces home advantage.")
    elif int(_get(row, "team_a_home_flag", 0) or 0):
        reasons.append(f"{a} has home advantage.")
    elif int(_get(row, "team_b_home_flag", 0) or 0):
        reasons.append(f"{b} has home advantage.")

    if not reasons:
        reasons.append("Teams look closely matched on the available signals.")
    return reasons[:4]


def detect_prediction_risks(row: pd.Series) -> list[str]:
    """Return reasons the prediction could be wrong (model-uncertainty framing)."""

    risks: list[str] = []
    competition = str(_get(row, "competition_type", "")).lower()
    sport = str(_get(row, "sport", "")).lower()

    if "friendly" in competition or "exhibition" in competition:
        if sport == "soccer":
            risks.append("International friendlies often have rotated, unpredictable lineups.")
        else:
            risks.append("Exhibition/friendly games are harder to model than competitive ones.")

    confidence_level = str(_get(row, "confidence_level") or assign_confidence_level(row))
    if confidence_level in {"Low", "Very low"}:
        risks.append("The outcome is close, so the model has low separation between teams.")

    prob_draw = float(_get(row, "prob_draw", 0.0) or 0.0)
    if prob_draw >= 0.27:
        risks.append("A draw is a realistic outcome here.")

    for warning in _availability_context(row):
        risks.append(warning)

    min_recent = _get(row, "min_recent_games")
    if min_recent is not None and int(min_recent) < 5:
        risks.append("At least one team has very few recent games to learn from.")

    congestion = float(_get(row, "schedule_congestion_diff", 0.0) or 0.0)
    if abs(congestion) >= 2:
        tired = _team_a(row) if congestion > 0 else _team_b(row)
        risks.append(f"{tired} is on a congested schedule, which adds variance.")

    if _get(row, "model_backtested", True) is False:
        risks.append("The model has not been backtested for this sport/league yet.")

    return risks


def detect_data_quality_warnings(row: pd.Series) -> list[str]:
    """Return concrete data-quality caveats to surface on the dashboard."""

    warnings: list[str] = []

    warnings.extend(_availability_context(row))
    a_out = int(_get(row, "team_a_key_players_out", 0) or 0)
    b_out = int(_get(row, "team_b_key_players_out", 0) or 0)
    if a_out:
        warnings.append(f"{_team_a(row)} has {a_out} key player(s) listed out.")
    if b_out:
        warnings.append(f"{_team_b(row)} has {b_out} key player(s) listed out.")

    a_recent = int(_get(row, "team_a_recent_games", 0) or 0)
    b_recent = int(_get(row, "team_b_recent_games", 0) or 0)
    if a_recent < 5:
        warnings.append(f"{_team_a(row)} has fewer than 5 recent games.")
    if b_recent < 5:
        warnings.append(f"{_team_b(row)} has fewer than 5 recent games.")

    venue = str(_get(row, "venue", "")).strip().lower()
    if venue in {"", "unknown", "nan"}:
        warnings.append("Venue is unknown.")

    competition = str(_get(row, "competition_type", "")).strip().lower()
    if competition in {"", "unknown", "nan"}:
        warnings.append("Competition type is unknown.")
    elif "friendly" in competition:
        warnings.append("Friendly match; lineups may be experimental.")

    if min(a_recent, b_recent) <= 0:
        warnings.append("Prediction uses fallback ratings only (no game history).")

    if _get(row, "model_backtested", True) is False:
        warnings.append("Model has not been backtested for this sport/league.")

    return warnings


def explain_prediction(row: pd.Series) -> dict:
    """Bundle confidence, reasons, risks, and warnings for one prediction row.

    Returns a dict with keys: ``predicted_outcome``, ``confidence_level``,
    ``data_quality``, ``key_reasons``, ``main_risks``,
    ``data_quality_warnings``.
    """

    data_quality = str(_get(row, "data_quality") or assign_prediction_data_quality(row))
    confidence_level = assign_confidence_level(row)
    # Make the derived labels available to the risk/warning detectors.
    enriched = dict(row) if not isinstance(row, dict) else row
    enriched["data_quality"] = data_quality
    enriched["confidence_level"] = confidence_level
    enriched_series = pd.Series(enriched)

    return {
        "predicted_outcome": str(_get(row, "predicted_outcome", "")),
        "confidence_level": confidence_level,
        "data_quality": data_quality,
        "key_reasons": _reasons(enriched_series),
        "main_risks": detect_prediction_risks(enriched_series),
        "data_quality_warnings": detect_data_quality_warnings(enriched_series),
    }
