"""Load and normalize future scheduled games (fixtures), no odds required.

Produces the canonical fixture schema consumed by the matchup feature builder:

    fixture_id, sport, league, game_date,
    team_a, team_b, team_a_home_flag, team_b_home_flag, neutral_site,
    competition_type, venue, status
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from data.sport_rules import normalize_sport
from data.team_name_map import apply_team_aliases, load_team_aliases


FIXTURE_COLUMNS: list[str] = [
    "fixture_id",
    "sport",
    "league",
    "game_date",
    "team_a",
    "team_b",
    "team_a_home_flag",
    "team_b_home_flag",
    "neutral_site",
    "competition_type",
    "venue",
    "status",
]


_COLUMN_SYNONYMS: dict[str, str] = {
    "fixture_id": "fixture_id",
    "game_id": "fixture_id",
    "match_id": "fixture_id",
    "id": "fixture_id",
    "sport": "sport",
    "league": "league",
    "competition": "league",
    "tournament": "league",
    "game_date": "game_date",
    "date": "game_date",
    "match_date": "game_date",
    "datetime": "game_date",
    "kickoff": "game_date",
    "competition_type": "competition_type",
    "match_type": "competition_type",
    "stage": "competition_type",
    "round": "competition_type",
    "team_a": "team_a",
    "team_b": "team_b",
    "home_team": "home_team",
    "home": "home_team",
    "home_team_name": "home_team",
    "away_team": "away_team",
    "away": "away_team",
    "away_team_name": "away_team",
    "neutral_site": "neutral_site",
    "neutral": "neutral_site",
    "team_a_home_flag": "team_a_home_flag",
    "team_b_home_flag": "team_b_home_flag",
    "venue": "venue",
    "stadium": "venue",
    "location": "venue",
    "status": "status",
}

_TRUE_STRINGS = {"1", "true", "yes", "y", "t"}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in _TRUE_STRINGS


def _build_column_lookup(df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in df.columns:
        role = _COLUMN_SYNONYMS.get(str(column).strip().lower())
        if role and role not in lookup:
            lookup[role] = column
    return lookup


def _make_fixture_id(row: pd.Series) -> str:
    raw = "|".join(
        str(row.get(field, ""))
        for field in ("game_date", "league", "team_a", "team_b")
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"fixture_{digest}"


def load_fixtures(path: str | Path) -> pd.DataFrame:
    """Load raw fixtures from a CSV file or a folder of CSV files."""

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
        raise FileNotFoundError(f"Fixtures file not found: {source}")
    try:
        return pd.read_csv(source, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_fixtures(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Map a raw fixtures frame onto the canonical fixture schema."""

    config = config or {}
    if df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    lookup = _build_column_lookup(df)
    out = pd.DataFrame(index=df.index)

    out["sport"] = (
        df[lookup["sport"]] if "sport" in lookup else config.get("default_sport", "soccer")
    )
    out["sport"] = out["sport"].map(normalize_sport)
    out["league"] = (
        df[lookup["league"]] if "league" in lookup else config.get("default_league", "unknown")
    )
    out["competition_type"] = (
        df[lookup["competition_type"]]
        if "competition_type" in lookup
        else config.get("default_competition_type", "unknown")
    )
    out["game_date"] = pd.to_datetime(
        df[lookup["game_date"]] if "game_date" in lookup else pd.NaT,
        errors="coerce",
    )
    out["venue"] = df[lookup["venue"]].astype(str) if "venue" in lookup else "unknown"
    out["status"] = df[lookup["status"]].astype(str) if "status" in lookup else "scheduled"

    neutral = (
        df[lookup["neutral_site"]].map(_coerce_bool)
        if "neutral_site" in lookup
        else pd.Series(False, index=df.index)
    )

    if "team_a" in lookup and "team_b" in lookup:
        out["team_a"] = df[lookup["team_a"]].astype(str)
        out["team_b"] = df[lookup["team_b"]].astype(str)
        if "team_a_home_flag" in lookup:
            a_home = df[lookup["team_a_home_flag"]].map(_coerce_bool)
        else:
            a_home = ~neutral
        b_home = (
            df[lookup["team_b_home_flag"]].map(_coerce_bool)
            if "team_b_home_flag" in lookup
            else pd.Series(False, index=df.index)
        )
    elif "home_team" in lookup and "away_team" in lookup:
        out["team_a"] = df[lookup["home_team"]].astype(str)
        out["team_b"] = df[lookup["away_team"]].astype(str)
        a_home = ~neutral
        b_home = pd.Series(False, index=df.index)
    else:
        raise ValueError(
            "Could not find team columns. Provide either team_a/team_b or "
            "home_team/away_team (case-insensitive)."
        )

    out["neutral_site"] = neutral.astype(int)
    out["team_a_home_flag"] = a_home.where(~neutral, False).astype(int)
    out["team_b_home_flag"] = b_home.where(~neutral, False).astype(int)

    aliases = load_team_aliases(config["aliases_path"]) if config.get("aliases_path") else {}
    out = apply_team_aliases(out, ["team_a", "team_b"], aliases)

    if "fixture_id" in lookup:
        out["fixture_id"] = df[lookup["fixture_id"]].astype(str)
    else:
        out["fixture_id"] = out.apply(_make_fixture_id, axis=1)

    out = out[FIXTURE_COLUMNS].copy()
    out = out.sort_values(["game_date", "fixture_id"]).reset_index(drop=True)
    return out


def validate_fixtures(df: pd.DataFrame) -> dict:
    """Return a structured validation report for a *normalized* fixtures frame."""

    issues: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in FIXTURE_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return {"ok": False, "n_rows": int(len(df)), "issues": issues, "warnings": warnings}

    if df.empty:
        warnings.append("No fixtures provided.")

    if df["fixture_id"].duplicated().any():
        issues.append(f"{int(df['fixture_id'].duplicated().sum())} duplicate fixture_id rows.")
    if df["game_date"].isna().any():
        warnings.append(f"{int(df['game_date'].isna().sum())} fixtures have an unparseable game_date.")
    unknown_venue = int((df["venue"].astype(str).str.lower().isin({"unknown", "", "nan"})).sum())
    if unknown_venue:
        warnings.append(f"{unknown_venue} fixtures have an unknown venue.")
    unknown_comp = int((df["competition_type"].astype(str).str.lower().isin({"unknown", "", "nan"})).sum())
    if unknown_comp:
        warnings.append(f"{unknown_comp} fixtures have an unknown competition_type.")

    dates = pd.to_datetime(df["game_date"], errors="coerce").dropna()
    return {
        "ok": len(issues) == 0,
        "n_rows": int(len(df)),
        "n_sports": int(df["sport"].nunique()),
        "n_leagues": int(df["league"].nunique()),
        "date_min": dates.min().isoformat() if not dates.empty else None,
        "date_max": dates.max().isoformat() if not dates.empty else None,
        "issues": issues,
        "warnings": warnings,
    }
