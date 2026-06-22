"""Data-quality gates for matchup predictions.

Unlike the old odds/CLV gates (which *blocked* signals), these gates never hide
a prediction unless it is genuinely impossible to score. Instead every fixture
is tagged with a quality level so the dashboard can show the prediction with an
honest warning. Levels, strongest to weakest:

    strong  -> plenty of recent games, fresh injuries, known context
    usable  -> enough to predict, minor caveats
    weak    -> thin history or several caveats; treat with caution
    very_weak -> almost no signal (e.g. both teams brand new)
"""

from __future__ import annotations

import pandas as pd

QUALITY_LEVELS = ["strong", "usable", "weak", "very_weak"]

_UNKNOWN_VALUES = {"", "unknown", "nan", "none"}


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


def _is_unknown(value) -> bool:
    return str(value).strip().lower() in _UNKNOWN_VALUES


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def assign_prediction_data_quality(row: pd.Series) -> str:
    """Return one of ``QUALITY_LEVELS`` for a single fixture prediction row."""

    n_recent = _get(row, "min_recent_games")
    if n_recent is None:
        n_recent = min(
            _get(row, "team_a_recent_games", 0) or 0,
            _get(row, "team_b_recent_games", 0) or 0,
        )
    n_recent = int(n_recent)

    # Impossible-to-score is the only hard floor: both teams have no history.
    if n_recent <= 0:
        return "very_weak"
    if n_recent < 5:
        return "weak"

    competition = str(_get(row, "competition_type", "unknown"))
    friendly = "friendly" in competition.lower()
    injury_present = bool(_get(row, "injury_data_present", 0))
    injury_stale = bool(_get(row, "team_a_injury_stale", True)) or bool(
        _get(row, "team_b_injury_stale", True)
    )
    availability_manual = _boolish(_get(row, "team_a_availability_manual", False)) or _boolish(
        _get(row, "team_b_availability_manual", False)
    )

    issues = 0
    if _is_unknown(competition):
        issues += 1
    if friendly:
        issues += 1
    if not injury_present or injury_stale:
        issues += 1
    if availability_manual:
        issues += 1

    if (
        n_recent >= 10
        and not _is_unknown(competition)
        and not friendly
        and injury_present
        and not injury_stale
        and not availability_manual
    ):
        return "strong"
    if issues >= 2:
        return "weak"
    return "usable"


def validate_matchup_training_data(df: pd.DataFrame) -> dict:
    """Validate a training feature frame; return a structured report."""

    issues: list[str] = []
    warnings: list[str] = []

    required = ["result_team_a_win", "result_draw", "result_team_b_win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing target columns: {missing}")
        return {"ok": False, "n_rows": int(len(df)), "issues": issues, "warnings": warnings}

    decided = (
        df["result_team_a_win"].astype(int)
        + df["result_draw"].astype(int)
        + df["result_team_b_win"].astype(int)
    ) == 1
    n_decided = int(decided.sum())
    if n_decided == 0:
        issues.append("No decided games in training data.")

    per_sport = {}
    if "sport" in df.columns:
        for sport, group in df.groupby("sport"):
            n = int(((group["result_team_a_win"] + group["result_draw"] + group["result_team_b_win"]) == 1).sum())
            per_sport[str(sport)] = n
            if n < 50:
                warnings.append(f"Sport '{sport}' has only {n} decided games; model may be weak.")

    draw_share = float(df["result_draw"].astype(int).mean()) if len(df) else 0.0

    return {
        "ok": len(issues) == 0,
        "n_rows": int(len(df)),
        "n_decided": n_decided,
        "n_undecided": int((~decided).sum()),
        "draw_share": round(draw_share, 4),
        "games_by_sport": per_sport,
        "issues": issues,
        "warnings": warnings,
    }


def validate_fixture_prediction_data(df: pd.DataFrame) -> dict:
    """Validate a fixture feature frame and summarize its data quality."""

    warnings: list[str] = []
    if df.empty:
        return {"ok": True, "n_fixtures": 0, "quality_counts": {}, "warnings": ["No fixtures."]}

    quality = df.apply(assign_prediction_data_quality, axis=1)
    counts = quality.value_counts().to_dict()
    counts = {level: int(counts.get(level, 0)) for level in QUALITY_LEVELS}

    low_history = int((df.get("min_recent_games", pd.Series(0, index=df.index)) < 5).sum())
    if low_history:
        warnings.append(f"{low_history} fixtures involve a team with fewer than 5 recent games.")

    if "injury_data_present" in df.columns:
        no_injuries = int((df["injury_data_present"].astype(int) == 0).sum())
        if no_injuries:
            warnings.append(f"{no_injuries} fixtures have no injury data.")
    if {"team_a_availability_manual", "team_b_availability_manual"} <= set(df.columns):
        manual_rows = int(
            (
                df["team_a_availability_manual"].map(_boolish)
                | df["team_b_availability_manual"].map(_boolish)
            ).sum()
        )
        if manual_rows:
            warnings.append(f"{manual_rows} fixtures use manual availability data.")

    if "competition_type" in df.columns:
        unknown_comp = int(df["competition_type"].map(_is_unknown).sum())
        if unknown_comp:
            warnings.append(f"{unknown_comp} fixtures have an unknown competition type.")

    return {
        "ok": True,
        "n_fixtures": int(len(df)),
        "quality_counts": counts,
        "warnings": warnings,
    }
