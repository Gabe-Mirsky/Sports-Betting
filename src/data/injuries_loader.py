"""Load and summarize injury / player-availability data (no odds required).

Canonical (normalized) columns produced by :func:`normalize_injuries`:

    team, player_name, status, injury_type, position,
    importance_score, expected_minutes_or_role, last_updated, return_estimate,
    source, notes

:func:`summarize_team_availability` rolls the player-level frame up to one row
per team, producing the availability features the matchup model consumes:
``team_injury_impact``, ``key_players_out``, ``players_out`` plus freshness
metadata used by the data-quality gates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.team_name_map import apply_team_aliases, load_team_aliases


INJURY_COLUMNS: list[str] = [
    "team",
    "player_name",
    "status",
    "injury_type",
    "position",
    "importance_score",
    "expected_minutes_or_role",
    "last_updated",
    "return_estimate",
    "source",
    "notes",
]


_COLUMN_SYNONYMS: dict[str, str] = {
    "team": "team",
    "team_name": "team",
    "club": "team",
    "player_name": "player_name",
    "player": "player_name",
    "name": "player_name",
    "status": "status",
    "availability": "status",
    "injury_status": "status",
    "injury_type": "injury_type",
    "injury": "injury_type",
    "ailment": "injury_type",
    "position": "position",
    "pos": "position",
    "importance_score": "importance_score",
    "importance": "importance_score",
    "impact": "importance_score",
    "expected_minutes_or_role": "expected_minutes_or_role",
    "role": "expected_minutes_or_role",
    "minutes": "expected_minutes_or_role",
    "usage": "expected_minutes_or_role",
    "last_updated": "last_updated",
    "updated": "last_updated",
    "report_date": "last_updated",
    "as_of": "last_updated",
    "return_estimate": "return_estimate",
    "return": "return_estimate",
    "expected_return": "return_estimate",
    "source": "source",
    "data_source": "source",
    "report_source": "source",
    "notes": "notes",
    "note": "notes",
    "comment": "notes",
    "comments": "notes",
}

# How much each status reduces availability (1.0 == fully unavailable).
_STATUS_WEIGHT: dict[str, float] = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.5,
    "probable": 0.2,
    "available": 0.0,
    "active": 0.0,
    "unknown": 0.5,
}

# Fallback importance by role keyword when no importance_score is supplied.
_ROLE_IMPORTANCE: list[tuple[tuple[str, ...], float]] = [
    (("star", "key", "franchise", "captain", "talisman"), 1.0),
    (("starter", "starting", "first team", "first-team"), 0.75),
    (("rotation", "squad"), 0.4),
    (("bench", "depth", "reserve", "backup"), 0.15),
]


def _normalize_status(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "unknown"
    # Map a few common phrasings onto canonical statuses.
    if text in {"gtd", "game-time decision", "game time decision"}:
        return "questionable"
    if text in {"inactive", "ruled out", "sidelined"}:
        return "out"
    if text in {"healthy", "no injury", "fit", "active"}:
        return "available"
    return text if text in _STATUS_WEIGHT else text


def _importance_from_role(role_text: object) -> float:
    text = str(role_text).strip().lower()
    if not text or text == "nan":
        return 0.25  # unknown
    for keywords, score in _ROLE_IMPORTANCE:
        if any(keyword in text for keyword in keywords):
            return score
    return 0.25


def _build_column_lookup(df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in df.columns:
        role = _COLUMN_SYNONYMS.get(str(column).strip().lower())
        if role and role not in lookup:
            lookup[role] = column
    return lookup


def load_injuries(path: str | Path) -> pd.DataFrame:
    """Load raw injuries from a CSV file or a folder of CSV files."""

    source = Path(path)
    if source.is_dir():
        frames = []
        for csv_path in sorted(source.glob("*.csv")):
            try:
                frames.append(pd.read_csv(csv_path, low_memory=False))
            except pd.errors.EmptyDataError:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not source.exists():
        raise FileNotFoundError(f"Injuries file not found: {source}")
    try:
        return pd.read_csv(source, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_injuries(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Map a raw injuries frame onto the canonical schema.

    Missing ``importance_score`` is filled from the role text using the simple
    fallback documented in the module docstring.
    """

    config = config or {}
    if df.empty:
        return pd.DataFrame(columns=INJURY_COLUMNS)

    lookup = _build_column_lookup(df)
    out = pd.DataFrame(index=df.index)

    out["team"] = df[lookup["team"]].astype(str) if "team" in lookup else ""
    out["player_name"] = df[lookup["player_name"]].astype(str) if "player_name" in lookup else ""
    out["status"] = (
        df[lookup["status"]].map(_normalize_status) if "status" in lookup else "unknown"
    )
    out["injury_type"] = df[lookup["injury_type"]].astype(str) if "injury_type" in lookup else "unknown"
    out["position"] = df[lookup["position"]].astype(str) if "position" in lookup else "unknown"
    out["expected_minutes_or_role"] = (
        df[lookup["expected_minutes_or_role"]].astype(str)
        if "expected_minutes_or_role" in lookup
        else "unknown"
    )
    out["last_updated"] = pd.to_datetime(
        df[lookup["last_updated"]] if "last_updated" in lookup else pd.NaT,
        errors="coerce",
    )
    out["return_estimate"] = (
        df[lookup["return_estimate"]].astype(str) if "return_estimate" in lookup else "unknown"
    )
    out["source"] = df[lookup["source"]].astype(str) if "source" in lookup else ""
    out["notes"] = df[lookup["notes"]].astype(str) if "notes" in lookup else ""

    if "importance_score" in lookup:
        importance = pd.to_numeric(df[lookup["importance_score"]], errors="coerce")
    else:
        importance = pd.Series(float("nan"), index=df.index, dtype="float64")
    # Fill missing importance from role text.
    fallback = out["expected_minutes_or_role"].map(_importance_from_role)
    out["importance_score"] = importance.fillna(fallback).clip(lower=0.0, upper=1.0)

    aliases = load_team_aliases(config["aliases_path"]) if config.get("aliases_path") else {}
    out = apply_team_aliases(out, ["team"], aliases)

    return out[INJURY_COLUMNS].copy()


def summarize_team_availability(
    injuries_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    stale_after_hours: float = 48.0,
) -> pd.DataFrame:
    """Roll injuries up to one row per team as of ``as_of_date``.

    Only reports updated on or before ``as_of_date`` are considered (so the
    summary never leaks future status changes when used inside the backtest).
    When ``last_updated`` is missing it is treated as always-applicable, and the
    team is flagged with ``injury_data_stale`` because we cannot verify
    freshness.

    Output columns: ``team``, ``team_injury_impact``, ``key_players_out``,
    ``players_out``, ``num_listed``, ``last_updated``, ``hours_since_update``,
    ``injury_data_stale``, ``availability_sources``, ``availability_notes``,
    ``availability_manual``.
    """

    columns = [
        "team",
        "team_injury_impact",
        "key_players_out",
        "players_out",
        "num_listed",
        "last_updated",
        "hours_since_update",
        "injury_data_stale",
        "availability_sources",
        "availability_notes",
        "availability_manual",
    ]
    if injuries_df is None or injuries_df.empty:
        return pd.DataFrame(columns=columns)

    as_of = pd.to_datetime(as_of_date, errors="coerce")
    frame = injuries_df.copy()
    frame["last_updated"] = pd.to_datetime(frame.get("last_updated"), errors="coerce")

    # Drop reports that post-date the as-of moment (avoid look-ahead).
    if as_of is not pd.NaT and not pd.isna(as_of):
        timed = frame["last_updated"].notna()
        frame = frame[~timed | (frame["last_updated"] <= as_of)].copy()

    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["importance_score"] = pd.to_numeric(
        frame.get("importance_score"), errors="coerce"
    ).fillna(0.25)
    status = frame["status"].astype(str).str.lower()
    frame["status_weight"] = status.map(_STATUS_WEIGHT).fillna(0.5)
    frame["row_impact"] = frame["importance_score"] * frame["status_weight"]
    frame["is_out"] = status.eq("out")
    frame["is_key_out"] = frame["is_out"] & (frame["importance_score"] >= 0.6)
    if "source" not in frame.columns:
        frame["source"] = ""
    if "notes" not in frame.columns:
        frame["notes"] = ""

    grouped = frame.groupby("team", dropna=False)
    summary = grouped.agg(
        team_injury_impact=("row_impact", "sum"),
        key_players_out=("is_key_out", "sum"),
        players_out=("is_out", "sum"),
        num_listed=("player_name", "count"),
        last_updated=("last_updated", "max"),
        availability_sources=("source", _join_unique_text),
        availability_notes=("notes", _join_unique_text),
    ).reset_index()

    summary["key_players_out"] = summary["key_players_out"].astype(int)
    summary["players_out"] = summary["players_out"].astype(int)

    if as_of is not pd.NaT and not pd.isna(as_of):
        hours = (as_of - summary["last_updated"]).dt.total_seconds() / 3600.0
    else:
        hours = pd.Series(pd.NA, index=summary.index, dtype="float64")
    summary["hours_since_update"] = hours
    # Stale if we can't verify freshness (no timestamp) or it's too old.
    summary["injury_data_stale"] = summary["hours_since_update"].isna() | (
        summary["hours_since_update"] > stale_after_hours
    )
    summary["availability_manual"] = summary["availability_sources"].astype(str).str.lower().str.contains("manual")
    return summary[columns]


def _join_unique_text(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        if text not in seen:
            seen.append(text)
    return "; ".join(seen)
